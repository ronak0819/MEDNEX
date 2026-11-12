from mednex.agent import MEDNEXAgent
from mednex.demo_cases import DemoCaseSet, load_demo_cases
from mednex.reporting import format_agent_report


def test_synthetic_demo_dataset_has_valid_named_examples() -> None:
    dataset = load_demo_cases()

    assert len(dataset.cases) >= 5
    assert "synthetic" in dataset.dataset_notice.casefold()
    assert "not clinical" in dataset.dataset_notice.casefold()
    assert len({case.id for case in dataset.cases}) == len(dataset.cases)


def test_demo_dataset_rejects_non_clinical_notice_missing() -> None:
    try:
        DemoCaseSet.model_validate(
            {
                "dataset_notice": "Example cases.",
                "cases": [
                    {
                        "id": "sample",
                        "title": "Sample",
                        "description": "A fictional person reports a cough.",
                        "learning_goal": "Exercise deterministic parsing.",
                    }
                ],
            }
        )
    except ValueError as error:
        assert "synthetic" in str(error)
    else:
        raise AssertionError("Dataset without synthetic notice was accepted.")


def test_complete_report_uses_actual_engine_outputs_and_safety_boundary() -> None:
    agent = MEDNEXAgent(max_follow_up_questions=5)
    turn = agent.start(
        "A fictional 45-year-old person has a cough and fever. "
        "Temperature is 38.5 degrees Celsius."
    )
    report = format_agent_report(turn)

    assert turn.bayesian is not None
    assert turn.ml_prediction is not None
    assert turn.symbolic is not None
    assert turn.graph is not None
    assert turn.information_gain is not None
    assert turn.bayesian.candidates[0].candidate_label in report
    assert turn.ml_prediction.predicted_label in report
    assert "Synthetic graph associations" in report
    assert "Ranked educational follow-up questions" in report
    assert "not clinical risk estimates" in report
    assert "does not recommend medicines, doses" in report
    assert "NOT FOR REAL PATIENT DECISIONS" in report


def test_safety_stopped_report_does_not_invent_engine_results() -> None:
    turn = MEDNEXAgent().start("I have a cough and fever.")
    report = format_agent_report(turn)

    assert turn.status == "PROFESSIONAL_ASSESSMENT_REQUIRED"
    assert turn.bayesian is None
    assert "Extraction and reasoning: Not run." in report
    assert "Safety boundary:" in report
    assert "Bayesian calculations" not in report
