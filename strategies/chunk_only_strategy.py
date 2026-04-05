import time
from typing import Any, Dict, List

from services.trace_service import append_trace
from strategies.base_strategy import BaseRetrievalStrategy


class ChunkOnlyStrategy(BaseRetrievalStrategy):
    strategy_id = "chunk_only"
    display_name = "Chunk Only"
    description = "Dense chunk retrieval only."

    async def run(self, *, question: str, dataset_name: str, graphq, kt_retriever, schema_path: str, options: Dict[str, Any], notify) -> Dict[str, Any]:
        sub_questions, _ = await self.decompose_question(
            question,
            graphq,
            schema_path,
            options.get("use_decomposition", True),
        )
        if notify:
            await notify({"stage": "decompose", "sub_questions_count": len(sub_questions)})

        trace: List[Dict[str, Any]] = []
        reasoning_steps: List[Dict[str, Any]] = []
        all_chunks: List[str] = []
        all_chunk_ids: List[str] = []

        for index, sub_question in enumerate(sub_questions):
            sub_text = sub_question.get("sub-question", question)
            started = time.perf_counter()
            retrieved = await self._run_blocking(
                kt_retriever.retrieve_chunk_only,
                sub_text,
                options.get("top_k", 20),
                None,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            all_chunks.extend(retrieved.get("chunk_contents", []))
            all_chunk_ids.extend(retrieved.get("chunk_ids", []))
            reasoning_steps.append({
                "type": "sub_question",
                "question": sub_text,
                "triples": [],
                "triples_count": 0,
                "chunks_count": len(retrieved.get("chunk_contents", [])),
                "processing_time": round(elapsed_ms / 1000, 3),
                "chunk_contents": retrieved.get("chunk_contents", [])[:3],
            })
            append_trace(
                trace,
                stage="retrieve",
                strategy_name=self.strategy_id,
                query=sub_text,
                sub_question_index=index,
                retrieved_chunks=retrieved.get("chunk_ids", []),
                duration_ms=elapsed_ms,
                meta={"path": "chunk_only"},
            )
            if notify:
                await notify({
                    "stage": "sub_question",
                    "index": index + 1,
                    "total": len(sub_questions),
                    "question": sub_text,
                    "triples_count": 0,
                    "chunks_count": len(retrieved.get("chunk_contents", [])),
                })

        final_chunks = self.dedup_preserve(all_chunks)[: options.get("top_k", 20)]
        answer = await self._run_blocking(
            kt_retriever.build_answer_from_context,
            question,
            [],
            final_chunks,
        )
        append_trace(
            trace,
            stage="answer",
            strategy_name=self.strategy_id,
            query=question,
            retrieved_chunks=all_chunk_ids,
            duration_ms=0.0,
        )

        return {
            "answer": answer,
            "sub_questions": sub_questions,
            "retrieved_triples": [],
            "retrieved_chunks": final_chunks,
            "reasoning_steps": reasoning_steps,
            "retrieval_trace": trace,
            "reasoning_subgraph": {"nodes": [], "links": [], "categories": []},
        }
