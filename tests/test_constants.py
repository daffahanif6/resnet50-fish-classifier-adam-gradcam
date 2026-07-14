"""Tests for the module-level constants exposed by ``fish``."""
import fish


def test_img_size_is_224_square():
    assert fish.IMG_SIZE == (224, 224)


def test_class_names_has_nine_unique_entries():
    assert len(fish.CLASS_NAMES) == 9
    assert len(set(fish.CLASS_NAMES)) == 9


def test_class_names_expected_values():
    assert fish.CLASS_NAMES == [
        "Black Sea Sprat",
        "Gilt-Head Bream",
        "Horse Mackerel",
        "Red Mullet",
        "Red Sea Bream",
        "Sea Bass",
        "Shrimp",
        "Striped Red Mullet",
        "Trout",
    ]


def test_max_file_size_and_min_model_size():
    assert fish.MAX_FILE_SIZE_MB == 5
    assert fish.MIN_MODEL_SIZE_BYTES == 1_000_000


def test_layer_candidates_non_empty_strings():
    assert fish.LAYER_CANDIDATES
    assert all(isinstance(name, str) and name for name in fish.LAYER_CANDIDATES)
