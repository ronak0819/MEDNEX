from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
from pydantic import Field, model_validator
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mednex.models import MEDNEXModel
from mednex.state import ExtractionResult

MODEL_NAMES = ("logistic_regression", "decision_tree", "random_forest")
SUPPORTED_SYMPTOMS = ("cough", "fever", "shortness of breath", "headache")
FEATURE_COLUMNS = {
    "cough": "cough",
    "fever": "fever",
    "shortness of breath": "shortness_of_breath",
    "headache": "headache",
}
VALID_STATES = {"present", "absent", "not_reported", "conflicting"}
RANDOM_SEED = 42
TEST_SIZE = 0.25
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "synthetic_ml_cases.csv"
)
METADATA_FILENAME = "mednex_ml_metadata.json"

DATASET_LIMITATIONS = [
    "All rows and labels are hand-authored synthetic teaching examples, not patient data.",
    "The candidate labels are arbitrary educational classes and are not diagnoses.",
    "Patterns in this small dataset were deliberately constructed and do not estimate real-world performance.",
    "The model uses only four supported symptoms; all other observations are ignored and reported.",
    "Unreported symptoms have an explicit not_reported feature state and are not treated as absent.",
    "The holdout scores describe only this synthetic split and must not be interpreted as clinical accuracy.",
    "Model probabilities are not calibrated clinical probabilities and are not combined with Bayesian outputs.",
]


class ClassifierMetrics(MEDNEXModel):
    model_name: str
    accuracy: float = Field(ge=0, le=1)
    precision_macro: float = Field(ge=0, le=1)
    recall_macro: float = Field(ge=0, le=1)
    f1_macro: float = Field(ge=0, le=1)
    class_labels: list[str]
    confusion_matrix: list[list[int]]


class ModelEvaluationReport(MEDNEXModel):
    dataset_path: str
    dataset_description: str
    total_samples: int = Field(gt=0)
    train_samples: int = Field(gt=0)
    test_samples: int = Field(gt=0)
    random_seed: int
    test_fraction: float = Field(gt=0, lt=1)
    stratified: bool
    metrics: list[ClassifierMetrics]
    limitations: list[str]


class MLPredictionResult(MEDNEXModel):
    model_name: str
    predicted_label: str
    class_probabilities: dict[str, float]
    supported_feature_states: dict[str, str]
    ignored_features: list[str]
    uncertainty_warning: str = (
        "This is an educational prediction from a small synthetic dataset. "
        "It is not a diagnosis, clinical probability, or recommendation."
    )
    separation_notice: str = (
        "These model outputs are separate from Bayesian inference and must not "
        "be blindly combined or averaged with Bayesian probabilities."
    )

    @model_validator(mode="after")
    def validate_class_probabilities(self) -> MLPredictionResult:
        if self.predicted_label not in self.class_probabilities:
            raise ValueError("Predicted label must have a corresponding class probability.")
        if any(not 0 <= probability <= 1 for probability in self.class_probabilities.values()):
            raise ValueError("Class probabilities must be between 0 and 1.")
        if not math.isclose(
            sum(self.class_probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("Class probabilities must sum to 1.")
        return self


def _feature_vector_from_extraction(
    extraction: ExtractionResult,
) -> tuple[dict[str, str], list[str]]:
    states: dict[str, set[bool]] = {name: set() for name in SUPPORTED_SYMPTOMS}
    ignored: set[str] = set()
    for symptom in extraction.state.symptoms:
        normalized = symptom.name.strip().casefold()
        if normalized not in states:
            ignored.add(f"symptom:{normalized}")
            continue
        states[normalized].add(symptom.is_negated)

    vector: dict[str, str] = {}
    for symptom, column in FEATURE_COLUMNS.items():
        polarities = states[symptom]
        if not polarities:
            value = "not_reported"
        elif len(polarities) > 1:
            value = "conflicting"
        elif True in polarities:
            value = "absent"
        else:
            value = "present"
        vector[column] = value

    for observation in extraction.state.extracted_observations:
        if observation.name != "symptom":
            ignored.add(f"observation:{observation.name}")
    if extraction.unsupported_statements:
        ignored.add("unparsed_unsupported_statements")
    if extraction.ambiguous_statements:
        ignored.add("ambiguous_statements")
    if extraction.contradictions:
        ignored.add("contradictory_input")
    return vector, sorted(ignored)


def _load_dataset(path: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    required_columns = {
        "dataset_notice",
        "label",
        *FEATURE_COLUMNS.values(),
    }
    with path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"Synthetic ML dataset is empty: {path}")
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Synthetic ML dataset is missing columns: {sorted(missing_columns)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Synthetic ML dataset has no samples: {path}")
    features: list[dict[str, str]] = []
    labels: list[str] = []
    notices: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        notice = row["dataset_notice"].strip()
        if "synthetic" not in notice.casefold():
            raise ValueError(
                f"Dataset row {line_number} is not labeled as synthetic."
            )
        notices.add(notice)
        feature_row = {
            column: row[column].strip().casefold()
            for column in FEATURE_COLUMNS.values()
        }
        invalid_values = {
            column: value
            for column, value in feature_row.items()
            if value not in VALID_STATES
        }
        if invalid_values:
            raise ValueError(
                f"Dataset row {line_number} has invalid feature states: {invalid_values}"
            )
        label = row["label"].strip()
        if not label.startswith("synthetic_"):
            raise ValueError(
                f"Dataset row {line_number} has a non-synthetic label: {label!r}"
            )
        features.append(feature_row)
        labels.append(label)
    if len(notices) != 1:
        raise ValueError("Synthetic ML dataset must use one consistent dataset notice.")
    return features, labels, list(notices)


def _make_pipeline(model_name: str, random_seed: int) -> Pipeline:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
        )
    elif model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=2,
            random_state=random_seed,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=2,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"Unsupported MEDNEX ML model: {model_name}")
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", classifier),
        ]
    )


class ClassicalMLEngine:
    """Train and run classical classifiers on the bundled synthetic cases only."""

    def __init__(
        self,
        models: dict[str, Pipeline],
        report: ModelEvaluationReport | None,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        if set(models) != set(MODEL_NAMES):
            raise ValueError(f"Models must include exactly: {MODEL_NAMES}")
        self.models = models
        self.report = report
        self.random_seed = random_seed

    @classmethod
    def train(
        cls,
        dataset_path: str | Path | None = None,
        random_seed: int = RANDOM_SEED,
        test_fraction: float = TEST_SIZE,
    ) -> ClassicalMLEngine:
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
        features, labels, notices = _load_dataset(path)
        class_counts = {label: labels.count(label) for label in sorted(set(labels))}
        if len(class_counts) < 2 or min(class_counts.values()) < 2:
            raise ValueError(
                "Stratified training requires at least two samples per class."
            )
        test_count = math.ceil(len(labels) * test_fraction)
        if test_count < len(class_counts) or len(labels) - test_count < len(class_counts):
            raise ValueError(
                "Train/test split must contain at least one sample from every class."
            )
        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            test_size=test_fraction,
            random_state=random_seed,
            stratify=labels,
        )
        class_labels = sorted(set(labels))
        models: dict[str, Pipeline] = {}
        metrics: list[ClassifierMetrics] = []
        for model_name in MODEL_NAMES:
            pipeline = _make_pipeline(model_name, random_seed)
            pipeline.fit(train_features, train_labels)
            predictions = pipeline.predict(test_features)
            matrix = confusion_matrix(
                test_labels, predictions, labels=class_labels
            )
            metrics.append(
                ClassifierMetrics(
                    model_name=model_name,
                    accuracy=accuracy_score(test_labels, predictions),
                    precision_macro=precision_score(
                        test_labels,
                        predictions,
                        labels=class_labels,
                        average="macro",
                        zero_division=0,
                    ),
                    recall_macro=recall_score(
                        test_labels,
                        predictions,
                        labels=class_labels,
                        average="macro",
                        zero_division=0,
                    ),
                    f1_macro=f1_score(
                        test_labels,
                        predictions,
                        labels=class_labels,
                        average="macro",
                        zero_division=0,
                    ),
                    class_labels=class_labels,
                    confusion_matrix=matrix.tolist(),
                )
            )
            models[model_name] = pipeline
        report = ModelEvaluationReport(
            dataset_path=str(path),
            dataset_description=(
                f"{len(features)} hand-authored synthetic examples tagged "
                f"{notices[0]!r}; labels are fictional educational classes."
            ),
            total_samples=len(features),
            train_samples=len(train_features),
            test_samples=len(test_features),
            random_seed=random_seed,
            test_fraction=test_fraction,
            stratified=True,
            metrics=metrics,
            limitations=DATASET_LIMITATIONS,
        )
        return cls(models=models, report=report, random_seed=random_seed)

    def predict(
        self,
        extraction: ExtractionResult,
        model_name: str = "logistic_regression",
    ) -> MLPredictionResult:
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model {model_name!r}; choose one of {MODEL_NAMES}."
            )
        feature_vector, ignored_features = _feature_vector_from_extraction(extraction)
        pipeline = self.models[model_name]
        predicted_label = str(pipeline.predict([feature_vector])[0])
        classifier: Any = pipeline.named_steps["classifier"]
        probabilities = classifier.predict_proba(
            pipeline.named_steps["vectorizer"].transform([feature_vector])
        )[0]
        class_probabilities = {
            str(label): float(probability)
            for label, probability in zip(classifier.classes_, probabilities)
        }
        return MLPredictionResult(
            model_name=model_name,
            predicted_label=predicted_label,
            class_probabilities=class_probabilities,
            supported_feature_states=feature_vector,
            ignored_features=ignored_features,
        )

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for model_name, pipeline in self.models.items():
            joblib.dump(pipeline, target / f"{model_name}.joblib")
        metadata = {
            "format_version": 1,
            "model_names": list(MODEL_NAMES),
            "random_seed": self.random_seed,
            "evaluation_report": (
                self.report.model_dump(mode="json") if self.report else None
            ),
            "warning": (
                "Only load MEDNEX model artifacts from trusted local sources. "
                "Predictions are educational outputs from synthetic data."
            ),
        }
        metadata_path = target / METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata_path

    @classmethod
    def load(cls, directory: str | Path) -> ClassicalMLEngine:
        target = Path(directory)
        metadata_path = target / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"MEDNEX model metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported MEDNEX model artifact format.")
        if set(metadata.get("model_names", [])) != set(MODEL_NAMES):
            raise ValueError("Model metadata does not list the expected model set.")
        models: dict[str, Pipeline] = {}
        for model_name in MODEL_NAMES:
            model_path = target / f"{model_name}.joblib"
            if not model_path.is_file():
                raise FileNotFoundError(f"MEDNEX model file not found: {model_path}")
            model = joblib.load(model_path)
            if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
                raise ValueError(f"Invalid MEDNEX model artifact: {model_path}")
            models[model_name] = model
        report_data = metadata.get("evaluation_report")
        report = ModelEvaluationReport.model_validate(report_data) if report_data else None
        return cls(
            models=models,
            report=report,
            random_seed=int(metadata["random_seed"]),
        )
