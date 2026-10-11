"""Entity detection and agreement scoring.

The case that matters: "SEM" is a substring of "SEM-IP", and reading a
passage about one as a passage about the other is the exact error this
module exists to prevent. Every genuine wrong answer in the phase 3
evaluation had that shape.
"""

from __future__ import annotations

from app.query.entities import agreement, detect


def test_sem_ip_is_not_read_as_sem():
    m = detect("Operating temperature of the SEM-IP is +5 to +40 degC.")
    assert m.by_family["room_module"] == {"SEM-IP"}
    assert "SEM" not in m.members()


def test_sem_is_detected_on_its_own():
    assert detect("The SEM terminals accept 0.6 mm wire.").by_family["room_module"] == {"SEM"}


def test_a_passage_mentioning_both_records_both():
    m = detect("The SEM and the SEM-IP differ in operating range.")
    assert m.by_family["room_module"] == {"SEM", "SEM-IP"}


def test_sem_plus_is_distinct_from_sem():
    m = detect("SEM+ supports audio.")
    assert "SEM+" in m.by_family["room_module"]
    assert m.by_family["room_module"] == {"SEM+"}


def test_r_abus_is_not_read_as_abus():
    m = detect("The R-ABUS supports up to 6 devices.")
    assert m.by_family["bus"] == {"R-ABUS"}
    assert "ABUS" not in m.members()


def test_buses_and_products_are_independent_families():
    m = detect("Connect the SEM to the DBUS.")
    assert m.by_family["room_module"] == {"SEM"}
    assert m.by_family["bus"] == {"DBUS"}


def test_spoken_aliases_resolve_to_the_product_code():
    assert "STIP" in detect("How do I mount the touch terminal?").members()
    assert "SCVIP" in detect("The Care View unit is a tabletop device.").members()


def test_agreement_rewards_the_matching_subject():
    q = detect("What temperature range can the SEM work in?")
    assert agreement(q, detect("The SEM operates from +5 to +55 degC.")) == 1.0


def test_agreement_penalises_a_sibling():
    """The SEM-IP page for a SEM question is evidence against, not
    merely absence of evidence. This is the signal retrieval lacked."""
    q = detect("What temperature range can the SEM work in?")
    assert agreement(q, detect("SEM-IP operating temperature +5 to +40 degC.")) == -1.0


def test_agreement_is_neutral_when_the_passage_names_nothing():
    q = detect("What temperature range can the SEM work in?")
    assert agreement(q, detect("Operating temperature +5 to +55 degC.")) == 0.0


def test_agreement_is_neutral_when_the_question_names_nothing():
    """Most questions name no entity. Those must not be penalised."""
    q = detect("What is the maximum screw head height?")
    assert agreement(q, detect("SEM-IP maximum screw head height 3 mm.")) == 0.0


def test_agreement_averages_across_families():
    q = detect("How many devices on the R-ABUS for the SEM?")
    # right bus, wrong module
    mixed = detect("The R-ABUS supports 6 devices on the SEM-IP.")
    assert agreement(q, mixed) == 0.0


def test_case_and_spacing_do_not_matter():
    assert "DBUS" in detect("the dbus run").members()
    assert "ZBUS" in detect("ZBUS: use a twisted wire").members()


def test_a_name_inside_a_longer_word_is_not_a_match():
    assert not detect("SEMANTIC search").members()
    assert not detect("ABUSE of the system").members()
