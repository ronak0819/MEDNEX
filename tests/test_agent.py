import pytest

from mednex.agent import MEDNEXAgent, _answer_polarity
from mednex.engines.information_engine import RankedQuestion


def _question(evidence_key: str) -> RankedQuestion:
    return RankedQuestion(
        question_id="synthetic-question",
        question="Synthetic follow-up?",
        evidence_key=evidence_key,
        information_gain_bits=0.1,
        current_entropy_bits=1,
        expected_posterior_entropy_bits=0.9,
        possible_answers=[],
        explanation="Synthetic test prompt.",
    )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("No", "absent"),
        ("not present", "absent"),
        ("Yes, present", "present"),
        ("I'm not sure", "unknown"),
    ],
)
def test_symptom_follow_up_answers_are_classified_explicitly(answer, expected) -> None:
    assert _answer_polarity(_question("symptom:cough"), answer) == expected


def test_measurement_follow_up_does_not_confuse_not_reported_with_reported() -> None:
    question = _question("observation:temperature_celsius")

    assert _answer_polarity(question, "not reported") == "not_reported"
    assert _answer_polarity(question, "Yes, it was reported") == "reported"


@pytest.fixture(scope="module")
def agent() -> MEDNEXAgent:
    return MEDNEXAgent(max_follow_up_questions=10)


def test_initial_agent_cycle_returns_one_follow_up_and_all_engine_outputs(agent) -> None:
    turn = agent.start("A fictional patient has cough.")

    assert turn.status == "NEEDS_MORE_INFORMATION"
    assert turn.question_to_user
    assert turn.session.current_question_id
    assert turn.extraction is not None
    assert turn.validation is not None and turn.validation.valid
    assert turn.graph is not None
    assert turn.symbolic is not None
    assert turn.bayesian is not None
    assert turn.ml_prediction is not None
    assert turn.ml_evaluation is not None
    assert turn.information_gain is not None
    assert turn.report.status == turn.status
    assert turn.report.disclaimer
    assert turn.session.current_question_id == turn.information_gain.ranked_questions[0].question_id


def test_follow_up_updates_state_and_explains_changes(agent) -> None:
    first = agent.start("A fictional patient has cough.")
    question_id = first.session.current_question_id
    assert question_id is not None

    second = agent.answer(first.session, "Yes")

    assert second.session.answers[question_id] == "Yes"
    assert question_id in second.session.answered_question_ids
    assert second.session.follow_up_count == 1
    assert second.session.current_question_id != question_id
    assert second.extraction is not None
    if question_id == "ask_fever":
        assert any(
            symptom.name == "fever" and not symptom.is_negated
            for symptom in second.extraction.state.symptoms
        )
    assert "Follow-up" in second.extraction.state.original_description
    assert second.change_explanation != first.change_explanation
    assert second.bayesian is not None
    assert second.symbolic is not None


def test_one_question_at_a_time_and_completed_session_closes(agent) -> None:
    turn = agent.start("A fictional patient has cough.")
    turns = 0
    while turn.status == "NEEDS_MORE_INFORMATION" and turns < 10:
        assert turn.question_to_user is not None
        assert turn.session.current_question_id is not None
        turn = agent.answer(turn.session, "Yes")
        turns += 1

    assert turn.status == "EVALUATION_COMPLETE"
    assert turn.session.closed
    assert turn.question_to_user is None
    assert "uncertainty" in turn.decision.next_action.lower()
    with pytest.raises(ValueError, match="session is closed"):
        agent.answer(turn.session, "Yes")


@pytest.mark.parametrize(
    "description",
    [
        "I have a cough and fever.",
        "A fictional patient has severe chest pain.",
        "A fictional patient is in an emergency.",
    ],
)
def test_safety_boundary_stops_evaluation(description, agent) -> None:
    turn = agent.start(description)

    assert turn.status == "PROFESSIONAL_ASSESSMENT_REQUIRED"
    assert turn.graph is None
    assert turn.bayesian is None
    assert turn.ml_prediction is None
    assert turn.safety.professional_assessment_required
    assert turn.report.disclaimer
    assert "qualified health professional" in turn.report.uncertainty_summary


def test_contradictions_produce_uncertain_stop_without_forcing_candidate(agent) -> None:
    turn = agent.start(
        "A fictional patient has fever but does not have a fever."
    )

    assert turn.status == "UNCERTAIN"
    assert turn.extraction is not None and turn.extraction.contradictions
    assert turn.bayesian is not None
    assert "fever" in turn.bayesian.contradictory_observations
    assert turn.question_to_user is None
    assert "Clarify" in turn.decision.next_action


def test_unsupported_text_with_known_symptom_stops_as_uncertain(agent) -> None:
    turn = agent.start(
        "A fictional patient has a cough and reports blurry vision."
    )

    assert turn.status == "UNCERTAIN"
    assert turn.extraction is not None
    assert turn.extraction.unsupported_statements
    assert turn.question_to_user is None
    assert "Unsupported" in " ".join(turn.report.missing_or_unmodeled)


def test_ml_is_skipped_when_case_has_no_supported_ml_features(agent) -> None:
    turn = agent.start("A fictional patient reports abdominal pain.")

    assert turn.ml_prediction is None
    assert turn.report.ml_summary is not None
    assert "was not run" in turn.report.ml_summary


def test_unknown_answer_is_recorded_but_not_invented_as_a_fact(agent) -> None:
    first = agent.start("A fictional patient has cough.")
    current_question_id = first.session.current_question_id
    assert current_question_id is not None

    second = agent.answer(first.session, "I'm not sure.")

    assert second.session.answers[current_question_id] == "I'm not sure."
    if current_question_id == "ask_fever":
        assert not any(
            symptom.name == "fever" for symptom in second.extraction.state.symptoms
        )
    assert second.extraction is not None


def test_symptom_correction_updates_extraction_and_reruns_engines(agent) -> None:
    first = agent.start("A fictional patient has cough.")

    corrected = agent.correct_symptoms(first.session, {"cough": "absent"})

    assert corrected.extraction is not None
    cough = [
        symptom
        for symptom in corrected.extraction.state.symptoms
        if symptom.name == "cough"
    ]
    assert len(cough) == 1
    assert cough[0].is_negated
    assert corrected.session.symptom_corrections == {"cough": "absent"}
    assert corrected.graph is not None
    assert corrected.symbolic is not None
    assert corrected.bayesian is not None
    assert corrected.ml_prediction is not None


def test_symptom_correction_can_remove_an_extracted_fact(agent) -> None:
    first = agent.start("A fictional patient has cough.")

    corrected = agent.correct_symptoms(first.session, {"cough": "not_reported"})

    assert corrected.extraction is not None
    assert not any(
        symptom.name == "cough" for symptom in corrected.extraction.state.symptoms
    )


def test_extracted_field_corrections_update_patient_and_vital_information(agent) -> None:
    first = agent.start(
        "A fictional 45-year-old male has cough. "
        "His temperature is 38.5 degrees Celsius."
    )

    corrected = agent.correct_extracted_fields(
        first.session,
        {"age_years": "46", "temperature_celsius": "37.2"},
    )

    assert corrected.extraction is not None
    assert corrected.extraction.state.patient_information.age_years == 46
    assert corrected.extraction.state.vital_signs is not None
    assert corrected.extraction.state.vital_signs.temperature_celsius == 37.2
    assert corrected.session.field_corrections == {
        "age_years": "46",
        "temperature_celsius": "37.2",
    }
    assert corrected.bayesian is not None
    assert corrected.session.current_question_id is not None

    continued = agent.answer(corrected.session, "unknown")

    assert continued.extraction is not None
    assert continued.extraction.state.patient_information.age_years == 46
    assert continued.extraction.state.vital_signs is not None
    assert continued.extraction.state.vital_signs.temperature_celsius == 37.2


def test_invalid_extracted_field_correction_is_reported(agent) -> None:
    first = agent.start("A fictional patient has cough.")

    with pytest.raises(ValueError, match="Unsupported extracted-field corrections"):
        agent.correct_extracted_fields(first.session, {"unsupported": "value"})

    with pytest.raises(ValueError):
        agent.correct_extracted_fields(
            first.session, {"temperature_celsius": "nan"}
        )


def test_follow_up_can_escalate_to_safety_boundary(agent) -> None:
    first = agent.start("A fictional patient has cough.")

    turn = agent.answer(first.session, "This is about me; I have the same symptoms.")

    assert turn.status == "PROFESSIONAL_ASSESSMENT_REQUIRED"
    assert turn.session.closed
    assert turn.graph is None


def test_invalid_empty_input_does_not_run_reasoning(agent) -> None:
    turn = agent.start("   ")

    assert turn.status == "UNCERTAIN"
    assert turn.validation is not None and not turn.validation.valid
    assert turn.graph is None
