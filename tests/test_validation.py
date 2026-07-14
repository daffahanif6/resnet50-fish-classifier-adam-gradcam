"""Tests for upload validation and simple UI helpers."""
from types import SimpleNamespace

import fish


def _uploaded(size_bytes):
    return SimpleNamespace(size=size_bytes, name="fish.png")


def test_validate_accepts_small_file():
    is_valid, error = fish.validate_uploaded_file(_uploaded(1024))
    assert is_valid is True
    assert error is None


def test_validate_accepts_file_at_limit():
    exactly_limit = fish.MAX_FILE_SIZE_MB * 1024 * 1024
    is_valid, error = fish.validate_uploaded_file(_uploaded(exactly_limit))
    assert is_valid is True
    assert error is None


def test_validate_rejects_oversized_file():
    too_big = (fish.MAX_FILE_SIZE_MB + 1) * 1024 * 1024
    is_valid, error = fish.validate_uploaded_file(_uploaded(too_big))
    assert is_valid is False
    assert error is not None
    assert str(fish.MAX_FILE_SIZE_MB) in error


class _FakeStreamlit:
    def __init__(self):
        self.images = []

    def image(self, path, **kwargs):
        self.images.append((path, kwargs))


def test_display_banner_shows_image_when_present(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(fish, "st", fake_st)
    monkeypatch.setattr(fish.os.path, "exists", lambda p: True)

    fish.display_banner()

    assert fake_st.images == [("banner.png", {"use_container_width": True})]


def test_display_banner_noop_when_missing(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(fish, "st", fake_st)
    monkeypatch.setattr(fish.os.path, "exists", lambda p: False)

    fish.display_banner()

    assert fake_st.images == []
