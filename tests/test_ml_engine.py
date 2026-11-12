import csv

import pytest

from mednex.engines.ml_engine import (
    MODEL_NAMES,
    ClassicalMLEngine,
    _load_dataset,
)
from mednex.nlp import parse_patient_description


@pytest.fixture(scope="module")
def trained_engine() -> ClassicalMLEngine:
    return ClassicalMLEngine.train()


def test_training_evaluates_all_required_models(trained_engine) -> None:
    assert set(trained_engine.models) == set(MODEL_NAMES)
    assert trained_engine.report is not None
    assert {metric.model_name for metric in trained_engine.report.metrics} == set(
        MODEL_NAMES
    )
    for metric in trained_engine.report.metrics:
        assert 0 <= metric.accuracy <= 1
        assert 0 <= metric.precision_macro <= 1
        assert 0 <= metric.recall_macro <= 1
        assert 0 <= metric.f1_macro <= 1
        assert len(metric.confusion_matrix) == len(metric.class_labels)
        assert all(
            len(row) == len(metric.class_labels)
            for row in metric.confusion_matrix
        )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_prediction_shape_probabilities_and_warning(trained_engine, model_name) -> None:
    result = trained_engine.predict(
        parse_patient_description(
            "A fictional patient has cough and fever."
        ),
        model_name=model_name,
    )

    assert result.model_name == model_name
    assert result.predicted_label in result.class_probabilities
    assert set(result.class_probabilities) == {
        "synthetic_pattern_alpha",
        "synthetic_pattern_beta",
        "synthetic_other",
    }
    assert sum(result.class_probabilities.values()) == pytest.approx(1)
    assert result.supported_feature_states["cough"] == "present"
    assert result.uncertainty_warning
    assert result.separation_notice


def test_missing_features_are_explicit_not_reported(trained_engine) -> None:
    result = trained_engine.predict(
        parse_patient_description("An unspecified fictional patient.")
    )

    assert set(result.supported_feature_states.values()) == {"not_reported"}


def test_negated_and_conflicting_symptoms_are_encoded_safely(trained_engine) -> None:
    negated = trained_engine.predict(
        parse_patient_description("A fictional patient has no cough.")
    )
    conflicting = trained_engine.predict(
        parse_patient_description("A fictional patient has cough but no cough.")
    )

    assert negated.supported_feature_states["cough"] == "absent"
    assert conflicting.supported_feature_states["cough"] == "conflicting"
    assert "contradictory_input" in conflicting.ignored_features


def test_unsupported_symptoms_and_vitals_are_reported_not_used(trained_engine) -> None:
    result = trained_engine.predict(
        parse_patient_description(
            "A fictional patient has dizziness. Temperature is 37 C."
        )
    )

    assert "symptom:dizziness" in result.ignored_features
    assert "observation:temperature_celsius" in result.ignored_features
    assert "dizziness" not in result.supported_feature_states


def test_training_is_reproducible(trained_engine) -> None:
    repeated = ClassicalMLEngine.train(random_seed=42)
    first = {
        item.model_name: item.model_dump()
        for item in trained_engine.report.metrics
    }
    second = {
        item.model_name: item.model_dump()
        for item in repeated.report.metrics
    }

    assert first == second


def test_save_and_load_models_preserves_predictions(trained_engine, tmp_path) -> None:
    trained_engine.save(tmp_path)
    loaded_engine = ClassicalMLEngine.load(tmp_path)
    extraction = parse_patient_description(
        "A fictional patient has shortness of breath and headache."
    )

    for model_name in MODEL_NAMES:
        before = trained_engine.predict(extraction, model_name)
        after = loaded_engine.predict(extraction, model_name)
        assert after.model_name == before.model_name
        assert after.predicted_label == before.predicted_label
        assert after.class_probabilities == pytest.approx(before.class_probabilities)


def test_loading_missing_model_metadata_fails_explicitly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="metadata not found"):
        ClassicalMLEngine.load(tmp_path)


def test_training_rejects_non_synthetic_dataset(tmp_path) -> None:
    bad_dataset = tmp_path / "cases.csv"
    with bad_dataset.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset_notice",
                "cough",
                "fever",
                "shortness_of_breath",
                "headache",
                "label",
            ]
        )
        writer.writerow(["observed data", "present", "absent", "absent", "absent", "real_label"])
        writer.writerow(["observed data", "absent", "present", "absent", "absent", "real_label"])

    with pytest.raises(ValueError, match="not labeled as synthetic"):
        _load_dataset(bad_dataset)


def test_evaluation_report_disclaims_clinical_performance(trained_engine) -> None:
    report = trained_engine.report

    assert report.stratified
    assert report.random_seed == 42
    assert report.train_samples + report.test_samples == report.total_samples
    assert any("do not estimate real-world performance" in item for item in report.limitations)
