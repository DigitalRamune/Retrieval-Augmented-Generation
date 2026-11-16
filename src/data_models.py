from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """Represent a retrieved document snippet with file path and character
    span."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a benchmark question without an answer."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class MinimalSearchResults(BaseModel):
    """Represent search results for a single question."""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent search results along with a generated answer for a
    question."""
    answer: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a ground-truth question with reference sources
    and answer."""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset containing answered or unanswered
      RAG questions."""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    """Represent batch search results for evaluation."""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent batch search results and generated answers."""
    search_results: list[MinimalAnswer]
    k: int
