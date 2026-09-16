"""Pruebas unitarias de src/models/load_model.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.load_model import (
    DEFAULT_CLASS_NAMES,
    ModelLoadError,
    get_class_names,
    load_model,
)


class TestLoadModel:
    def test_raises_when_file_missing(self, tmp_path):
        missing_path = tmp_path / "no_such_model.pt"
        load_model.cache_clear()

        with pytest.raises(ModelLoadError):
            load_model(str(missing_path))

    def test_error_message_mentions_path(self, tmp_path):
        missing_path = tmp_path / "ghost.pt"
        load_model.cache_clear()

        with pytest.raises(ModelLoadError, match="ghost.pt"):
            load_model(str(missing_path))

    def test_load_model_raises_when_model_cannot_be_loaded(self, tmp_path):
        model_path = tmp_path / "invalid_model.pt"
        model_path.write_text("not a real YOLO model")

        load_model.cache_clear()

        with pytest.raises(ModelLoadError):
            load_model(str(model_path))


class TestGetClassNames:
    def test_returns_default_when_model_is_none(self):
        assert get_class_names(None) == DEFAULT_CLASS_NAMES

    def test_returns_model_names_when_available(self):
        model = MagicMock()
        model.names = {0: "helmet", 1: "gloves"}

        assert get_class_names(model) == ("helmet", "gloves")

    def test_falls_back_to_default_when_names_empty(self):
        model = MagicMock()
        model.names = {}

        assert get_class_names(model) == DEFAULT_CLASS_NAMES

    def test_returns_model_names_from_list(self):
        model = MagicMock()
        model.names = ["helmet", "gloves", "vest"]

        assert get_class_names(model) == ("helmet", "gloves", "vest")

    def test_returns_model_names_from_tuple(self):
        model = MagicMock()
        model.names = ("helmet", "gloves", "vest")

        assert get_class_names(model) == ("helmet", "gloves", "vest")

    @pytest.mark.parametrize(
        "expected_class",
        [
            "Gloves",
            "Vest",
            "Goggles",
            "helmet",
            "Mask",
            "safety_shoe",
        ],
    )
    def test_default_class_names_contains_expected(self, expected_class):
        assert expected_class in DEFAULT_CLASS_NAMES