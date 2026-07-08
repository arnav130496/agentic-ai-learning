from pydantic import BaseModel, Field

class State(BaseModel):
    query: str = Field(description="User Query")
    retrieved_chunks:list[dict] = Field(description="Each dict object contains chunk_id, text, rerank_score, source")
    confidence_score: float = Field(description="Confidence score of the summary", default=None)
    evidence_sufficient: bool = Field(description="Whether the retrieved evidence is sufficient to answer the query", default=None)
    answer: str = Field(description="Answer to the user query", default=None)
    citations: list[str] = Field(description="List of cited sources", default=None)

State.model_json_schema()

class ConstrainedAnswer(BaseModel):
    answer: str = Field(description="Answer grounded only in the provided context")

ConstrainedAnswer.model_json_schema()