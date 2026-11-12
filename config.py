from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str = "MEDNEX"
    version: str = "0.1.0"
    educational_notice: str = (
        "Educational simulation only; not a diagnosis or a substitute for professional care."
    )


CONFIG = ProjectConfig()
