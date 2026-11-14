"""Output classes of the wildfire model and the rule for labelling images.

The model outputs 3 probabilities: [p_fire, p_smoke, p_neither]. Everything
that produces labels or reads model output must use this module so the
position of each class is the same everywhere (training, metrics, ONNX
inference on the satellite).

Plain Python only (no numpy/tensorflow) so it can be imported anywhere.
"""
import operator

FIRE = "fire"
SMOKE = "smoke"
NEITHER = "neither"

# Order = model output order. Tuple so it cannot be changed at runtime.
# Deliberately not alphabetical: sklearn's LabelEncoder would sort to
# (fire, neither, smoke) and silently swap two classes.
CLASSES = (FIRE, SMOKE, NEITHER)
NUM_CLASSES = len(CLASSES)

_LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}


def label_to_index(label: str) -> int:
    """Return the model output position of a class label.

    Strict: only the exact strings in CLASSES are accepted. Normalising
    free text (case, whitespace, synonyms) belongs to the caller.

    Raises:
        ValueError: if the label is not one of CLASSES.
    """
    try:
        return _LABEL_TO_INDEX[label]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown class label {label!r}; expected one of {CLASSES}"
        ) from None


def index_to_label(index: int) -> str:
    """Return the class label at a model output position.

    Accepts Python and numpy integers (e.g. the result of np.argmax).

    Raises:
        ValueError: if the index is not an integer in [0, NUM_CLASSES).
            Negative indices are rejected instead of wrapping around.
    """
    try:
        position = operator.index(index)
    except TypeError:
        raise ValueError(f"Class index must be an integer, got {index!r}") from None
    if isinstance(index, bool) or not 0 <= position < NUM_CLASSES:
        raise ValueError(
            f"Class index {index!r} out of range; expected 0..{NUM_CLASSES - 1}"
        )
    return CLASSES[position]


def resolve_label(fire_visible: bool, smoke_visible: bool) -> str:
    """Collapse what is visible in an image into exactly one class.

    Priority: fire > smoke > neither.
      - fire:    active fire visible, with or without smoke
      - smoke:   smoke visible, no fire
      - neither: no fire and no smoke (clouds, haze, clear ground, ...)

    Raises:
        TypeError: if either argument is not a real bool. Prevents parsed
            text like "no" (truthy) from being read as True.
    """
    for name, value in (("fire_visible", fire_visible), ("smoke_visible", smoke_visible)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool, got {type(value).__name__}: {value!r}")

    if fire_visible:
        return FIRE
    if smoke_visible:
        return SMOKE
    return NEITHER
