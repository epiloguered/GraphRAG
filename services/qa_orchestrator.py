import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from config import get_config
from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever
from services.result_formatter import build_comparison_summary, format_strategy_result
from services.strategy_registry import StrategyRegistry


class QAOrchestrator:
    def __init__(self, manager, get_schema_path):
        self.manager = manager
        self.get_schema_path = get_schema_path
        self.registry = StrategyRegistry()
        self.config = None

    def list_modes(self) -> List[Dict[str, str]]:
        return self.registry.list_modes()

    async def _send_message(self, client_id: str, payload: Dict[str, Any]) -> None:
        if client_id:
            await self.manager.send_message(payload, client_id)

    async def _send_progress(self, client_id: str, stage: str, progress: int, message: str) -> None:
        await self._send_message(client_id, {
            "type": "progress",
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    def _ensure_config(self):
        if self.config is None:
            self.config = get_config("config/base_config.yaml")
        return self.config

    def _model_to_dict(self, model) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def _resolve_graph_path(self, dataset_name: str) -> str:
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if not os.path.exists(graph_path):
            raise FileNotFoundError("Graph not found. Please construct graph first.")
        return graph_path

    async def _prepare_components(self, dataset_name: str):
        config = self._ensure_config()
        schema_path = self.get_schema_path(dataset_name)
        graph_path = self._resolve_graph_path(dataset_name)
        graphq = decomposer.GraphQ(dataset_name, config=config)
        kt_retriever = retriever.KTRetriever(
            dataset_name,
            graph_path,
            recall_paths=config.retrieval.recall_paths,
            schema_path=schema_path,
            top_k=config.retrieval.top_k_filter,
            mode="agent",
            config=config,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, kt_retriever.build_indices)
        return graphq, kt_retriever, schema_path

    def _write_experiment_record(self, dataset_name: str, question: str, request_options: Dict[str, Any], result: Dict[str, Any]) -> None:
        os.makedirs("output/experiments", exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "question": question,
            "strategy_name": result.get("strategy_name"),
            "request_options": request_options,
            "answer": result.get("answer"),
            "sub_questions": result.get("sub_questions", []),
            "retrieved_triples": result.get("retrieved_triples", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "metrics": result.get("metrics", {}),
            "retrieval_trace": result.get("retrieval_trace", []),
        }
        output_path = os.path.join("output", "experiments", f"{dataset_name}.jsonl")
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def run_single(self, request, client_id: str = "default") -> Dict[str, Any]:
        strategy_name = request.retrieval_mode or "youtu_default"
        await self._send_progress(client_id, "retrieval", 10, "Initializing retrieval system...")
        graphq, kt_retriever, schema_path = await self._prepare_components(request.dataset_name)
        await self._send_progress(client_id, "retrieval", 35, "Indices ready. Running strategy...")
        strategy = self.registry.get(strategy_name)
        request_options = self._model_to_dict(request)

        async def notify(payload: Dict[str, Any]) -> None:
            await self._send_message(client_id, {
                "type": "qa_update",
                "strategy_name": strategy_name,
                "timestamp": datetime.now().isoformat(),
                **payload,
            })

        started = time.perf_counter()
        raw_result = await strategy.run(
            question=request.question,
            dataset_name=request.dataset_name,
            graphq=graphq,
            kt_retriever=kt_retriever,
            schema_path=schema_path,
            options=request_options,
            notify=notify,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        formatted = format_strategy_result(
            raw_result,
            strategy_name=strategy_name,
            latency_ms=latency_ms,
            decomposition_applied=request.use_decomposition,
            ircot_applied=(strategy_name == "youtu_default" and request.use_ircot),
            ircot_steps=raw_result.get("ircot_steps", 0) if strategy_name == "youtu_default" and request.use_ircot else 0,
            retrieval_rounds=len([step for step in raw_result.get("retrieval_trace", []) if step.get("stage") == "retrieve"]),
        )
        self._write_experiment_record(request.dataset_name, request.question, request_options, formatted)
        await self._send_progress(client_id, "retrieval", 100, "Answer generation completed!")
        await self._send_message(client_id, {
            "type": "qa_complete",
            "strategy_name": strategy_name,
            "answer_preview": (formatted.get("answer") or "")[:300],
            "sub_questions_count": len(formatted.get("sub_questions", [])),
            "triples_final_count": len(formatted.get("retrieved_triples", [])),
            "chunks_final_count": len(formatted.get("retrieved_chunks", [])),
            "timestamp": datetime.now().isoformat(),
        })
        return formatted

    async def run_compare(self, request, client_id: str = "default") -> Dict[str, Any]:
        compare_modes = list(dict.fromkeys(request.compare_modes or []))
        if not compare_modes:
            raise ValueError("compare_modes must not be empty")
        request_options = self._model_to_dict(request)

        await self._send_progress(client_id, "compare", 10, "Initializing comparison...")
        graphq, kt_retriever, schema_path = await self._prepare_components(request.dataset_name)
        await self._send_progress(client_id, "compare", 30, "Indices ready. Running strategies...")

        results = []
        total = len(compare_modes)
        for index, strategy_name in enumerate(compare_modes):
            strategy = self.registry.get(strategy_name)

            async def notify(payload: Dict[str, Any], current_strategy: str = strategy_name) -> None:
                await self._send_message(client_id, {
                    "type": "compare_update",
                    "strategy_name": current_strategy,
                    "timestamp": datetime.now().isoformat(),
                    **payload,
                })

            await notify({"stage": "start", "progress": 0, "message": f"Running {strategy_name}"})
            started = time.perf_counter()
            raw_result = await strategy.run(
                question=request.question,
                dataset_name=request.dataset_name,
                graphq=graphq,
                kt_retriever=kt_retriever,
                schema_path=schema_path,
                options=request_options,
                notify=notify,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            formatted = format_strategy_result(
                raw_result,
                strategy_name=strategy_name,
                latency_ms=latency_ms,
                decomposition_applied=request.use_decomposition,
                ircot_applied=(strategy_name == "youtu_default" and request.use_ircot),
                ircot_steps=raw_result.get("ircot_steps", 0) if strategy_name == "youtu_default" and request.use_ircot else 0,
                retrieval_rounds=len([step for step in raw_result.get("retrieval_trace", []) if step.get("stage") == "retrieve"]),
            )
            self._write_experiment_record(request.dataset_name, request.question, request_options, formatted)
            results.append(formatted)
            await notify({"stage": "complete", "progress": 100, "message": f"{strategy_name} completed"})
            await self._send_progress(
                client_id,
                "compare",
                min(95, 30 + int(((index + 1) / total) * 60)),
                f"Completed {index + 1}/{total} strategies",
            )

        payload = {
            "question": request.question,
            "dataset_name": request.dataset_name,
            "results": results,
            "comparison_summary": build_comparison_summary(results),
        }
        await self._send_progress(client_id, "compare", 100, "Comparison completed!")
        await self._send_message(client_id, {
            "type": "compare_complete",
            "results_count": len(results),
            "timestamp": datetime.now().isoformat(),
        })
        return payload
