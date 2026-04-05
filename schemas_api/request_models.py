from typing import List

from pydantic import BaseModel, Field


class AskQuestionRequest(BaseModel):
    question: str
    dataset_name: str
    retrieval_mode: str = "youtu_default"
    top_k: int = Field(default=20, ge=1, le=100)
    use_decomposition: bool = True
    use_ircot: bool = True
    max_steps: int = Field(default=3, ge=1, le=10)
    return_trace: bool = True
    return_subgraph: bool = True


class CompareQuestionRequest(BaseModel):
    question: str
    dataset_name: str
    compare_modes: List[str]
    top_k: int = Field(default=20, ge=1, le=100)
    use_decomposition: bool = True
    use_ircot: bool = False
    max_steps: int = Field(default=3, ge=1, le=10)
    return_trace: bool = True
    return_subgraph: bool = True
