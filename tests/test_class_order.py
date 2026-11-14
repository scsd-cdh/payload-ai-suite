"""Contract for the model's output classes.

The model outputs 3 probabilities. Every part of the pipeline (labels,
training, metrics, ONNX inference on the satellite) must agree on which
position means which class. These tests pin that order down.
"""
import pytest

from classes import CLASSES, NUM_CLASSES, index_to_label, label_to_index


def test_classes_are_fire_smoke_neither_in_that_order():
    assert CLASSES == ("fire", "smoke", "neither")


def test_classes_cannot_be_modified():
    # A tuple, so no code can append/reorder classes at runtime.
    assert isinstance(CLASSES, tuple)


def test_num_classes_matches_class_list():
    assert NUM_CLASSES == 3
    assert NUM_CLASSES == len(CLASSES)


@pytest.mark.parametrize(
    "label, expected_index",
    [("fire", 0), ("smoke", 1), ("neither", 2)],
)
def test_label_to_index(label, expected_index):
    assert label_to_index(label) == expected_index


@pytest.mark.parametrize(
    "index, expected_label",
    [(0, "fire"), (1, "smoke"), (2, "neither")],
)
def test_index_to_label(index, expected_label):
    assert index_to_label(index) == expected_label


@pytest.mark.parametrize("label", ["fire", "smoke", "neither"])
def test_label_round_trip(label):
    assert index_to_label(label_to_index(label)) == label


def test_order_is_not_alphabetical():
    # sklearn's LabelEncoder sorts alphabetically (fire, neither, smoke),
    # which would silently swap smoke and neither. Guard against that.
    assert list(CLASSES) != sorted(CLASSES)


@pytest.mark.parametrize(
    "bad_label",
    ["cloud", "", "yes", "no", "Fire", " fire", "FIRE"],
)
def test_unknown_or_unnormalized_label_raises(bad_label):
    # Strict on purpose: cleaning up text (case, spaces) is the job of
    # whatever parses Gemini/human input, not of this contract.
    with pytest.raises(ValueError):
        label_to_index(bad_label)


@pytest.mark.parametrize("bad_index", [-1, 3, 99])
def test_out_of_range_index_raises(bad_index):
    # -1 matters: plain tuple indexing would silently return "neither".
    with pytest.raises(ValueError):
        index_to_label(bad_index)
