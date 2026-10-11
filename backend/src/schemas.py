from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    db_type: str = Field(..., description="postgresql | mysql | sqlserver")
    db_schema: str = Field(default="", description="Схема БД в виде строки")


class MissingDimension(BaseModel):
    code: str
    description: str


class AnalyzeResponse(BaseModel):
    ambiguity_status: str
    completeness_score: float
    missing_dimensions: list[MissingDimension] = []
    reasons: list[str] = []
    suggested_clarifications: list[str] = []
