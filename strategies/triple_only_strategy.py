import time
from typing import Any, Dict, List

from services.trace_service import append_trace
from strategies.base_strategy import BaseRetrievalStrategy


class TripleOnlyStrategy(BaseRetrievalStrategy):
    strategy_id = "triple_only"
    display_name = "Triple Only"
    description = "FAISS triple and community retrieval."

    async def run(self, *, question: str, dataset_name: str, graphq, kt_retriever, schema_path: str, options: Dict[str, Any], notify) -> Dict[str, Any]:
        sub_questions, involved_types = await self.decompose_question(
            question,
            graphq,
            schema_path,
            options.get("use_decomposition", True),
        )
        if notify:
            await notify({"stage": "decompose", "sub_questions_count": len(sub_questions)})

        trace: List[Dict[str, Any]] = []
        reasoning_steps: List[Dict[str, Any]] = []
        all_triples: List[str] = []
        all_chunks: List[str] = []

        for index, sub_question in enumerate(sub_questions):
            sub_text = sub_question.get("sub-question", question)
            started = time.perf_counter()
            retrieved = await self._run_blocking(
                kt_retriever.retrieve_triple_only,
                sub_text,
                options.get("top_k", 20),
                involved_types,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            all_triples.extend(retrieved.get("triples", []))
            all_chunks.extend(retrieved.get("chunk_contents", []))
            reasoning_steps.append({
                "type": "sub_question",
                "question": sub_text,
                "triples": retrieved.get("triples", [])[:10],
                "triples_count": len(retrieved.get("triples", [])),
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
                retrieved_triples=retrieved.get("triples", [])[:10],
                retrieved_chunks=retrieved.get("chunk_ids", []),
                retrieved_nodes=retrieved.get("retrieved_nodes", []),
                duration_ms=elapsed_ms,
                meta={"path": "triple_only", "structured_triples": retrieved.get("structured_triples", [])},
            )
            if notify:
                await notify({
                    "stage": "sub_question",
                    "index": index + 1,
                    "total": len(sub_questions),
                    "question": sub_text,
                    "triples_count": len(retrieved.get("triples", [])),
                    "chunks_count": len(retrieved.get("chunk_contents", [])),
                })

        final_triples = self.dedup_preserve(all_triples)[: options.get("top_k", 20)]
        final_chunks = self.dedup_preserve(all_chunks)[: options.get("top_k", 20)]
        answer = await self._run_blocking(
            kt_retriever.build_answer_from_context,
            question,
            final_triples,
            final_chunks,
        )
        subgraph = await self._run_blocking(
            kt_retriever.build_reasoning_subgraph,
            final_triples,
            trace,
        )
        append_trace(
            trace,
            stage="answer",
            strategy_name=self.strategy_id,
            query=question,
            retrieved_triples=final_triples[:10],
        )

        return {
            "answer": answer,
            "sub_questions": sub_questions,
            "retrieved_triples": final_triples,
            "retrieved_chunks": final_chunks,
            "reasoning_steps": reasoning_steps,
            "retrieval_trace": trace,
            "reasoning_subgraph": subgraph,
        }
