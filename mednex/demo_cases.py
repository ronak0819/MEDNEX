from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, field_validator

from mednex.models import MEDNEXModel

DEFAULT_DEMO_CASES_PATH = Path(__file__).resolve().parent / "data" / "demo_cases.json"


class DemoCase(MEDNEXModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    learning_goal: str = Field(min_length=1)


class DemoCaseSet(MEDNEXModel):
    dataset_notice: str = Field(min_length=1)
    cases: list[DemoCase] = Field(min_length=1)

    @field_validator("dataset_notice")
    @classmethod
    def require_synthetic_notice(cls, value: str) -> str:
        normalized = value.casefold()
        if "synthetic" not in normalized or "not clinical" not in normalized:
            raise ValueError(
                "Demo-case dataset must explicitly identify itself as synthetic "
                "and not clinical."
            )
        return value

    @field_validator("cases")
    @classmethod
    def require_unique_case_ids(cls, cases: list[DemoCase]) -> list[DemoCase]:
        identifiers = [case.id for case in cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Demo case identifiers must be unique.")
        return cases


def load_demo_cases(path: str | Path | None = None) -> DemoCaseSet:
    source_path = Path(path) if path is not None else DEFAULT_DEMO_CASES_PATH
    with source_path.open("r", encoding="utf-8") as data_file:
        return DemoCaseSet.model_validate(json.load(data_file))
