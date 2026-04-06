from typing import Any, Dict, List

from pydantic import BaseModel, Field


class StrategyRunResponse(BaseModel):
    """Normalized response returned by both single-run and compare-run flows."""

    strategy_name: str
    answer: str
    sub_questions: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_triples: List[str] = Field(default_factory=list)
    retrieved_chunks: List[str] = Field(default_factory=list)
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_trace: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_subgraph: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)


class CompareQuestionResponse(BaseModel):
    """Response returned by the strategy comparison endpoint."""

    question: str
    dataset_name: str
    results: List[StrategyRunResponse] = Field(default_factory=list)
    comparison_summary: Dict[str, Any] = Field(default_factory=dict)
