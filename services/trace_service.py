from typing import Any, Dict, List, Optional


def create_trace_step(
    step_id: int,
    stage: str,
    strategy_name: str,
    query: str,
    sub_question_index: Optional[int] = None,
    retrieved_triples: Optional[List[str]] = None,
    retrieved_chunks: Optional[List[str]] = None,
    retrieved_nodes: Optional[List[str]] = None,
    duration_ms: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "step_id": step_id,
        "stage": stage,
        "strategy_name": strategy_name,
        "query": query,
        "sub_question_index": sub_question_index,
        "retrieved_triples": retrieved_triples or [],
        "retrieved_chunks": retrieved_chunks or [],
        "retrieved_nodes": retrieved_nodes or [],
        "duration_ms": round(duration_ms, 3),
        "meta": meta or {},
    }


def append_trace(
    trace: List[Dict[str, Any]],
    stage: str,
    strategy_name: str,
    query: str,
    sub_question_index: Optional[int] = None,
    retrieved_triples: Optional[List[str]] = None,
    retrieved_chunks: Optional[List[str]] = None,
    retrieved_nodes: Optional[List[str]] = None,
    duration_ms: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    step = create_trace_step(
        step_id=len(trace) + 1,
        stage=stage,
        strategy_name=strategy_name,
        query=query,
        sub_question_index=sub_question_index,
        retrieved_triples=retrieved_triples,
        retrieved_chunks=retrieved_chunks,
        retrieved_nodes=retrieved_nodes,
        duration_ms=duration_ms,
        meta=meta,
    )
    trace.append(step)
    return step
