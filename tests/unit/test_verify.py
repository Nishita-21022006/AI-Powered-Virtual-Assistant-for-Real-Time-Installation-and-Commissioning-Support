"""Tier 3 verification checks.

Every number and part code in an answer appears verbatim in a cited
source; a citation outside the retrieved set is rejected; a procedure
answered without its warnings is rejected; an answer with no citation is
never shown. Deterministic checks, so these are exact-assertion tests.
"""

import dataclasses

import pytest

from app.query.verify import Rejection, SourcePassage, verify

TORQUE = SourcePassage(
    doc_id="INST-4412",
    text=(
        "4.2 Mounting the bracket\n"
        "Step 3: Tighten the four M6 mounting screws to 2.5 Nm.\n"
        "Step 4: Fit the cover plate. Maximum screw head height is 3 mm.\n"
    ),
    section="4.2",
    page=12,
)

WARNED = SourcePassage(
    doc_id="INST-4412",
    text=(
        "5.1 Replacing the control board\n"
        "Step 1: Remove the terminal cover.\n"
        "Step 2: Unplug connector X7 from the board.\n"
    ),
    section="5.1",
    page=18,
    warnings=("WARNING: Isolate the supply and verify zero volts before removing the cover.",),
)

CATALOG = SourcePassage(
    doc_id="CAT-2026",
    text="Replacement control board, part number E-412. Compatible fuse: F05, 6.3 A.",
    page=4,
)


def test_accepts_answer_whose_figures_and_codes_are_all_in_the_cited_source():
    verdict = verify("Tighten the M6 screws to 2.5 Nm. [INST-4412]", [TORQUE])
    assert verdict.accepted
    assert verdict.reasons == ()
    assert verdict.cited_doc_ids == ("INST-4412",)
    assert bool(verdict) is True


def test_uncited_answer_is_rejected_even_when_every_value_is_correct():
    verdict = verify("Tighten the M6 screws to 2.5 Nm.", [TORQUE])
    assert not verdict.accepted
    assert verdict.reasons == (Rejection.NO_CITATION,)


def test_empty_answer_is_rejected():
    assert verify("   ", [TORQUE]).reasons == (Rejection.NO_CITATION,)


def test_answer_with_no_retrieved_passages_at_all_is_rejected():
    verdict = verify("Tighten to 2.5 Nm. [INST-4412]", [])
    assert not verdict.accepted
    assert Rejection.NO_CITATION in verdict.reasons
    assert Rejection.CITATION_NOT_RETRIEVED in verdict.reasons


def test_invented_citation_is_rejected():
    verdict = verify("Tighten to 2.5 Nm. [INST-9999]", [TORQUE])
    assert not verdict.accepted
    assert Rejection.CITATION_NOT_RETRIEVED in verdict.reasons
    assert any("INST-9999" in d for d in verdict.details)


def test_one_real_citation_does_not_excuse_a_second_invented_one():
    verdict = verify("Tighten to 2.5 Nm. [INST-4412] See also [INST-9999]", [TORQUE])
    assert not verdict.accepted
    assert verdict.reasons == (Rejection.CITATION_NOT_RETRIEVED,)
    assert verdict.cited_doc_ids == ("INST-4412",)


def test_citation_with_page_and_section_label_still_resolves():
    verdict = verify("Tighten to 2.5 Nm. [INST-4412, section 4.2, p. 12]", [TORQUE])
    assert verdict.accepted


def test_semicolon_separated_multi_document_citation_resolves_both():
    answer = "Board E-412 is tightened to 2.5 Nm. [INST-4412; CAT-2026]"
    verdict = verify(answer, [TORQUE, CATALOG])
    assert verdict.accepted
    assert verdict.cited_doc_ids == ("CAT-2026", "INST-4412")


def test_citation_resolution_is_case_insensitive():
    assert verify("Tighten to 2.5 Nm. [inst-4412]", [TORQUE]).accepted


def test_structured_citation_list_is_honoured_without_inline_markers():
    verdict = verify("Tighten to 2.5 Nm.", [TORQUE], cited_doc_ids=["INST-4412"])
    assert verdict.accepted


def test_structured_citation_list_outside_retrieved_set_is_rejected():
    verdict = verify("Tighten to 2.5 Nm.", [TORQUE], cited_doc_ids=["INST-0001"])
    assert Rejection.CITATION_NOT_RETRIEVED in verdict.reasons


def test_number_found_nowhere_is_rejected_although_the_document_is_real():
    verdict = verify("Tighten the M6 screws to 4.5 Nm. [INST-4412]", [TORQUE])
    assert not verdict.accepted
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)
    assert any("4.5 Nm" in d for d in verdict.details)


def test_number_present_only_in_an_uncited_retrieved_passage_is_rejected():
    verdict = verify("The fuse is rated 6.3 A. [INST-4412]", [TORQUE, CATALOG])
    assert not verdict.accepted
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)


def test_fabricated_part_code_is_rejected():
    verdict = verify("Order replacement board E-519. [CAT-2026]", [CATALOG])
    assert not verdict.accepted
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)
    assert any("E-519" in d for d in verdict.details)


def test_fabricated_fault_code_is_rejected():
    verdict = verify("Fault F11 indicates a blown fuse. [CAT-2026]", [CATALOG])
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)


def test_fabricated_terminal_label_is_rejected():
    verdict = verify("Unplug connector X9 from the board. [INST-4412]", [WARNED])
    assert Rejection.UNSUPPORTED_VALUE in verdict.reasons


@pytest.mark.parametrize(
    "answer_form",
    ["3 mm", "3mm", "3 MM", "3.0 mm", "3,0 mm", "03 mm", "3.00mm", "3 millimetres"],
)
def test_formatting_variance_on_a_correct_figure_is_accepted(answer_form):
    answer = "Maximum screw head height is " + answer_form + ". [INST-4412]"
    assert verify(answer, [TORQUE]).accepted


@pytest.mark.parametrize("answer_form", ["2.5 Nm", "2,5 Nm", "2.50 N.m", "2.5Nm"])
def test_decimal_separator_and_unit_spelling_variance_is_accepted(answer_form):
    answer = "Tighten to " + answer_form + ". [INST-4412]"
    assert verify(answer, [TORQUE]).accepted


def test_thousands_separator_variance_is_accepted():
    source = SourcePassage("SPEC-1", "Rated speed 1500 rpm at 1000 Hz.")
    assert verify("Rated speed is 1,500 rpm at 1,000 Hz. [SPEC-1]", [source]).accepted


def test_a_near_miss_number_is_still_rejected_after_normalisation():
    source = SourcePassage("SPEC-1", "Rated speed 1500 rpm.")
    verdict = verify("Rated speed is 1,550 rpm. [SPEC-1]", [source])
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)


def test_identifier_separator_variance_is_accepted():
    unwarned = SourcePassage("SPEC-1", "Unplug connector X7 and fuse F05.")
    assert verify("Order board E412. [CAT-2026]", [CATALOG]).accepted
    assert verify("Unplug connector X-7 and fuse F-05. [SPEC-1]", [unwarned]).accepted


def test_unit_swapped_for_one_absent_from_the_source_is_rejected():
    verdict = verify("Maximum screw head height is 3 inches. [INST-4412]", [TORQUE])
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)
    assert any("absent" in d for d in verdict.details)


def test_sign_inversion_is_rejected():
    source = SourcePassage("SPEC-1", "Operating range 10 degC to 40 degC.")
    verdict = verify("The lower limit is -10 degC. [SPEC-1]", [source])
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)


def test_range_notation_does_not_lose_its_upper_bound():
    source = SourcePassage("SPEC-1", "Analogue input accepts 0-10 V.")
    assert verify("The input accepts 0-10 V. [SPEC-1]", [source]).accepted
    assert not verify("The input accepts 0-24 V. [SPEC-1]", [source]).accepted


def test_part_code_is_not_decomposed_into_a_bare_number_that_matches_elsewhere():
    source = SourcePassage("SPEC-1", "Set parameter P12 to 412. Board part number is E-808.")
    verdict = verify("Order board E-412. [SPEC-1]", [source])
    assert verdict.reasons == (Rejection.UNSUPPORTED_VALUE,)


def test_spelled_out_number_against_a_numeric_source_is_not_matched_and_so_accepts():
    assert verify("Tighten the four screws evenly. [INST-4412]", [TORQUE]).accepted


def test_procedure_answer_that_silently_drops_its_warning_is_rejected():
    answer = "Remove the terminal cover, then unplug connector X7. [INST-4412]"
    verdict = verify(answer, [WARNED])
    assert not verdict.accepted
    assert verdict.reasons == (Rejection.OMITTED_WARNING,)
    assert any("Isolate the supply" in d for d in verdict.details)


def test_procedure_answer_carrying_the_warning_is_accepted():
    answer = (
        "Isolate the supply and verify zero volts before removing the cover. "
        "Then remove the terminal cover and unplug connector X7. [INST-4412]"
    )
    assert verify(answer, [WARNED]).accepted


def test_partially_carried_warning_below_the_coverage_floor_is_rejected():
    answer = "Isolate the supply, then unplug connector X7. [INST-4412]"
    assert verify(answer, [WARNED]).reasons == (Rejection.OMITTED_WARNING,)


def test_warning_whose_figure_is_dropped_is_rejected_even_if_the_words_survive():
    passage = SourcePassage(
        "INST-4412",
        "Step 4: Fit the cover plate.",
        warnings=("WARNING: Maximum screw head height is 3 mm or the cover will foul.",),
    )
    answer = (
        "Fit the cover plate. Maximum screw head height must be respected "
        "or the cover will foul. [INST-4412]"
    )
    verdict = verify(answer, [passage])
    assert verdict.reasons == (Rejection.OMITTED_WARNING,)


def test_warning_restated_with_its_figure_in_a_different_format_is_accepted():
    passage = SourcePassage(
        "INST-4412",
        "Step 4: Fit the cover plate.",
        warnings=("WARNING: Maximum screw head height is 3 mm or the cover will foul.",),
    )
    answer = (
        "Fit the cover plate. Maximum screw head height is 3.0mm, or the cover will foul. "
        "[INST-4412]"
    )
    assert verify(answer, [passage]).accepted


def test_warning_on_an_uncited_passage_is_not_demanded():
    answer = "Order replacement board E-412. [CAT-2026]"
    assert verify(answer, [WARNED, CATALOG]).accepted


def test_every_failing_check_is_reported_not_just_the_first():
    passage = SourcePassage(
        "INST-4412",
        "Step 2: Unplug connector X7.",
        warnings=("WARNING: Isolate the supply and verify zero volts first.",),
    )
    answer = "Unplug connector X7 after loosening the 9 Nm clamp. [INST-4412] [MAN-7]"
    verdict = verify(answer, [passage])
    assert set(verdict.reasons) == {
        Rejection.CITATION_NOT_RETRIEVED,
        Rejection.UNSUPPORTED_VALUE,
        Rejection.OMITTED_WARNING,
    }


def test_an_unrecognised_bracketed_span_counts_as_an_invented_citation():
    verdict = verify("Tighten to 2.5 Nm [INST-4412]. See [Figure 3].", [TORQUE])
    assert Rejection.CITATION_NOT_RETRIEVED in verdict.reasons


def test_values_inside_citation_markers_are_not_treated_as_answer_claims():
    verdict = verify("Tighten to 2.5 Nm. [INST-4412, p. 12]", [TORQUE])
    assert verdict.accepted


def test_verdict_is_immutable():
    verdict = verify("Tighten to 2.5 Nm. [INST-4412]", [TORQUE])
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.accepted = False


def test_verify_does_not_mutate_its_inputs():
    passages = [WARNED, CATALOG]
    before = [(p.doc_id, p.text, p.warnings) for p in passages]
    verify("Order board E-412. [CAT-2026]", passages)
    assert [(p.doc_id, p.text, p.warnings) for p in passages] == before
