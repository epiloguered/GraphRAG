import re
import time
from typing import Any, Dict, List

from services.trace_service import append_trace
from strategies.base_strategy import BaseRetrievalStrategy


class YoutuDefaultStrategy(BaseRetrievalStrategy):
    """Default strategy that mirrors the original Youtu GraphRAG flow."""

    strategy_id = "youtu_default"
    display_name = "Youtu Default"
    description = "Decomposition + hybrid retrieval + IRCoT."

    async def run(self, *, question: str, dataset_name: str, graphq, kt_retriever, schema_path: str, options: Dict[str, Any], notify) -> Dict[str, Any]:
        """Run decomposition, hybrid retrieval, and optional IRCoT refinement."""
        sub_questions, involved_types = await self.decompose_question(
            question,
            graphq,
            schema_path,
            options.get("use_decomposition", True),
        )
        trace: List[Dict[str, Any]] = []
        reasoning_steps: List[Dict[str, Any]] = []
        if notify:
            await notify({"stage": "decompose", "sub_questions_count": len(sub_questions)})
        append_trace(
            trace,
            stage="decompose",
            strategy_name=self.strategy_id,
            query=question,
            meta={"sub_questions_count": len(sub_questions)},
        )

        all_triples: List[str] = []
        all_chunks: List[str] = []

        # Retrieve evidence independently for each decomposed sub-question first.
        for index, sub_question in enumerate(sub_questions):
            sub_text = sub_question.get("sub-question", question)
            started = time.perf_counter()
            retrieved = await self._run_blocking(
                kt_retriever.retrieve_youtu_hybrid,
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
                meta={"path": "youtu_hybrid", "structured_triples": retrieved.get("structured_triples", [])},
            )
            if notify:
                await notify({
                    "stage": "sub_question",
                    "index": index + 1,
                    "total": len(sub_questions),
                    "question": sub_text,
                    "triples_count": len(retrieved.get("triples", [])),
                    "chunks_count": len(retrieved.get("chunk_contents", [])),
                    "triples_preview": retrieved.get("triples", [])[:5],
                })

        final_triples = self.dedup_preserve(all_triples)[: options.get("top_k", 20)]
        final_chunks = self.dedup_preserve(all_chunks)[: options.get("top_k", 20)]
        initial_answer = await self._run_blocking(
            kt_retriever.build_answer_from_context,
            question,
            final_triples,
            final_chunks,
        )
        final_answer = initial_answer
        thoughts = [f"Initial: {initial_answer[:200]}"] if initial_answer else []
        ircot_steps = 0

        # Optional IRCoT loop expands the evidence set with follow-up retrieval queries.
        if options.get("use_ircot", True):
            current_query = question
            for step in range(1, options.get("max_steps", 3) + 1):
                ircot_steps = step
                loop_ctx = "=== Triples ===\n" + "\n".join(final_triples[:20])
                loop_ctx += "\n=== Chunks ===\n" + "\n---\n".join(final_chunks[:10])
                loop_prompt = f"""
You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.
Current Question: {question}
Current Iteration Query: {current_query}
Knowledge Context:
{loop_ctx}
Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}
Instructions:
1. If enough info answer with: So the answer is: <answer>
2. Else propose new query with: The new query is: <query>
Your reasoning:
"""
                reasoning = await self._run_blocking(kt_retriever.generate_answer, loop_prompt)
                thoughts.append((reasoning or "")[:400])
                reasoning_steps.append({
                    "type": "ircot_step",
                    "question": current_query,
                    "triples": final_triples[:10],
                    "triples_count": len(final_triples),
                    "chunks_count": len(final_chunks),
                    "processing_time": 0,
                    "chunk_contents": final_chunks[:3],
                    "thought": (reasoning or "")[:300],
                })
                append_trace(
                    trace,
                    stage="iterate",
                    strategy_name=self.strategy_id,
                    query=current_query,
                    retrieved_triples=final_triples[:10],
                    retrieved_chunks=final_chunks[:10],
                    meta={"thought": (reasoning or "")[:300]},
                )
                if notify:
                    await notify({
                        "stage": "ircot",
                        "step": step,
                        "max_steps": options.get("max_steps", 3),
                        "current_query": current_query,
                        "thought_preview": (reasoning or "")[:200],
                    })
                if "So the answer is:" in (reasoning or ""):
                    match = re.search(r"So the answer is:\s*(.*)", reasoning, flags=re.IGNORECASE | re.DOTALL)
                    final_answer = match.group(1).strip() if match else reasoning
                    break
                if "The new query is:" not in (reasoning or ""):
                    break
                new_query = reasoning.split("The new query is:", 1)[1].strip().splitlines()[0]
                if not new_query or new_query == current_query:
                    break
                current_query = new_query
                retrieved = await self._run_blocking(
                    kt_retriever.retrieve_youtu_hybrid,
                    current_query,
                    options.get("top_k", 20),
                    involved_types,
                )
                final_triples = self.dedup_preserve(final_triples + retrieved.get("triples", []))[: options.get("top_k", 20)]
                final_chunks = self.dedup_preserve(final_chunks + retrieved.get("chunk_contents", []))[: options.get("top_k", 20)]
                append_trace(
                    trace,
                    stage="retrieve",
                    strategy_name=self.strategy_id,
                    query=current_query,
                    retrieved_triples=retrieved.get("triples", [])[:10],
                    retrieved_chunks=retrieved.get("chunk_ids", []),
                    retrieved_nodes=retrieved.get("retrieved_nodes", []),
                    meta={"path": "ircot_followup", "structured_triples": retrieved.get("structured_triples", [])},
                )

        # Final trace drives both the response payload and the reasoning subgraph UI.
        append_trace(
            trace,
            stage="answer",
            strategy_name=self.strategy_id,
            query=question,
            retrieved_triples=final_triples[:10],
            retrieved_chunks=final_chunks[:10],
        )
        subgraph = await self._run_blocking(
            kt_retriever.build_reasoning_subgraph,
            final_triples,
            trace,
        )

        return {
            "answer": final_answer,
            "sub_questions": sub_questions,
            "retrieved_triples": final_triples,
            "retrieved_chunks": final_chunks,
            "reasoning_steps": reasoning_steps,
            "retrieval_trace": trace,
            "reasoning_subgraph": subgraph,
            "ircot_steps": ircot_steps,
        }
