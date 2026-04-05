from typing import Any, Dict, List

from services.visualization_service import prepare_strategy_visualization


def build_metrics(
    result: Dict[str, Any],
    *,
    latency_ms: float,
    decomposition_applied: bool,
    ircot_applied: bool,
    ircot_steps: int,
    retrieval_rounds: int,
) -> Dict[str, Any]:
    return {
        "latency_ms": round(latency_ms, 3),
        "triples_count": len(result.get("retrieved_triples", [])),
        "chunks_count": len(result.get("retrieved_chunks", [])),
        "sub_questions_count": len(result.get("sub_questions", [])),
        "retrieval_rounds": retrieval_rounds,
        "ircot_applied": ircot_applied,
        "ircot_steps": ircot_steps,
        "decomposition_applied": decomposition_applied,
    }


def format_strategy_result(
    raw_result: Dict[str, Any],
    *,
    strategy_name: str,
    latency_ms: float,
    decomposition_applied: bool,
    ircot_applied: bool,
    ircot_steps: int,
    retrieval_rounds: int,
) -> Dict[str, Any]:
    result = {
        "strategy_name": strategy_name,
        "answer": raw_result.get("answer", ""),
        "sub_questions": raw_result.get("sub_questions", []),
        "retrieved_triples": raw_result.get("retrieved_triples", []),
        "retrieved_chunks": raw_result.get("retrieved_chunks", []),
        "reasoning_steps": raw_result.get("reasoning_steps", []),
        "retrieval_trace": raw_result.get("retrieval_trace", []),
        "reasoning_subgraph": raw_result.get("reasoning_subgraph", {}),
    }
    result["metrics"] = build_metrics(
        result,
        latency_ms=latency_ms,
        decomposition_applied=decomposition_applied,
        ircot_applied=ircot_applied,
        ircot_steps=ircot_steps,
        retrieval_rounds=retrieval_rounds,
    )
    result["visualization_data"] = prepare_strategy_visualization(result)
    return result


def build_comparison_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    fastest = min(results, key=lambda item: item.get("metrics", {}).get("latency_ms", float("inf")))
    most_triples = max(results, key=lambda item: item.get("metrics", {}).get("triples_count", 0))
    most_chunks = max(results, key=lambda item: item.get("metrics", {}).get("chunks_count", 0))

    return {
        "strategies_count": len(results),
        "fastest_strategy": fastest.get("strategy_name"),
        "most_triples_strategy": most_triples.get("strategy_name"),
        "most_chunks_strategy": most_chunks.get("strategy_name"),
    }
