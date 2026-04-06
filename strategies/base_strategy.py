import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseRetrievalStrategy(ABC):
    """Shared helpers for all retrieval strategies."""

    strategy_id = "base"
    display_name = "Base"
    description = ""

    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking retriever or LLM call without stalling the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def decompose_question(
        self,
        question: str,
        graphq,
        schema_path: str,
        use_decomposition: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Optionally decompose a question and return a stable fallback shape on failure."""
        if not use_decomposition:
            return [{"sub-question": question}], {"nodes": [], "relations": [], "attributes": []}

        try:
            decomposition = await self._run_blocking(graphq.decompose, question, schema_path)
            sub_questions = decomposition.get("sub_questions", []) or [{"sub-question": question}]
            involved_types = decomposition.get("involved_types", {}) or {
                "nodes": [],
                "relations": [],
                "attributes": [],
            }
            return sub_questions, involved_types
        except Exception:
            # Strategies should remain usable even when decomposition fails.
            return [{"sub-question": question}], {"nodes": [], "relations": [], "attributes": []}

    def dedup_preserve(self, items: List[Any]) -> List[Any]:
        """Deduplicate while preserving order for strings, dicts, and trace payloads."""
        seen = set()
        output = []
        for item in items:
            key = repr(item)
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    @abstractmethod
    async def run(
        self,
        *,
        question: str,
        dataset_name: str,
        graphq,
        kt_retriever,
        schema_path: str,
        options: Dict[str, Any],
        notify,
    ) -> Dict[str, Any]:
        """Execute the strategy and return raw retrieval and reasoning artifacts."""
        raise NotImplementedError
