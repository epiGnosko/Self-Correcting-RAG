from pydantic import BaseModel, Field

class ReflectionResult(BaseModel):
    grounded_in_context: bool = Field(
        description="Whether the answer is supported by the provided context."
    )

    fully_answered: bool = Field(
        description="Whether all parts of the user's question are answered."
    )

    hallucination_detected: bool = Field(
        description="Whether the answer contains unsupported claims."
    )

    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score."
    )

    knowledge_gap: bool = Field(
        description="True if additional retrieval is needed."
    )

    reasoning: str = Field(
        description="Brief explanation of the evaluation."
    )

    verdict: str = Field(
        description="good | regenerate | retrieve_more"
    )
