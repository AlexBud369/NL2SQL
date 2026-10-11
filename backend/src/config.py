from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
        frozen=True,
    )

    project_name: str = Field("ambigues for NL2SQL service", validation_alias="PROJECT_NAME")
    project_version: str = Field("1.0", validation_alias="PROJECT_VERSION")

    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")

    host: str = Field(..., validation_alias="HOST")
    port: int = Field(..., validation_alias="PORT")

    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=["*"], validation_alias="CORS_ORIGINS"
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value):
        if not isinstance(value, str):
            return value

        trimmed_input = value.strip()

        if trimmed_input == "":
            return []

        raw_pieces = trimmed_input.split(",")

        cleaned_pieces = []
        for piece in raw_pieces:
            cleaned = piece.strip()
            if cleaned:
                cleaned_pieces.append(cleaned)

        return cleaned_pieces


settings = Settings()
