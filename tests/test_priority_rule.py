"""Contract for turning "what is visible" into exactly one class.

Softmax needs classes that never overlap. Fire and smoke usually appear
together, so the rule is: fire wins over smoke, smoke wins over neither.
  - fire:    active fire visible (with or without smoke)
  - smoke:   smoke visible, no fire
  - neither: no fire and no smoke (clouds, haze, clear ground, ...)
"""
import pytest

from classes import CLASSES, resolve_label


@pytest.mark.parametrize(
    "fire_visible, smoke_visible, expected",
    [
        (True, True, "fire"),
        (True, False, "fire"),
        (False, True, "smoke"),
        (False, False, "neither"),
    ],
)
def test_priority_rule_truth_table(fire_visible, smoke_visible, expected):
    assert resolve_label(fire_visible, smoke_visible) == expected


@pytest.mark.parametrize("fire_visible", [True, False])
@pytest.mark.parametrize("smoke_visible", [True, False])
def test_result_is_always_a_known_class(fire_visible, smoke_visible):
    assert resolve_label(fire_visible, smoke_visible) in CLASSES


@pytest.mark.parametrize(
    "fire_visible, smoke_visible",
    [
        ("no", False),   # non-empty string is truthy -> would become "fire"
        (False, "yes"),
        (1, False),
        (None, False),
    ],
)
def test_non_boolean_input_raises(fire_visible, smoke_visible):
    # Inputs will come from parsed Gemini answers. A raw string like "no"
    # must not slip through as True, so only real bools are accepted.
    with pytest.raises(TypeError):
        resolve_label(fire_visible, smoke_visible)
