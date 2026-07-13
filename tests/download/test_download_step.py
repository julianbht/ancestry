"""Tests for the ingest step's source selection and strict config schema."""

import pytest
from pydantic import ValidationError

from pipeline.download.config import DownloadConfig
from pipeline.download.step import _build_source
from pipeline.download.sources.local import LocalSource
from pipeline.download.sources.nextcloud import NextcloudSource
from pipeline.download.sources.r2 import R2Source


def _cfg(source: str) -> DownloadConfig:
    return DownloadConfig(source=source, max_files_to_download=None, ignore_state=False)


@pytest.mark.parametrize(
    "source,expected",
    [("local", LocalSource), ("r2", R2Source)],
)
def test_build_source_returns_matching_impl(source, expected):
    assert isinstance(_build_source(_cfg(source)), expected)


def test_build_source_nextcloud(monkeypatch):
    # nextcloud is the only branch needing external config; stub load_shares so
    # the test doesn't depend on the (private) shares.json.
    monkeypatch.setattr("pipeline.download.step.load_shares", lambda: object())
    assert isinstance(_build_source(_cfg("nextcloud")), NextcloudSource)


def test_config_rejects_unknown_source():
    with pytest.raises(ValidationError):
        _cfg("dropbox")


def test_config_rejects_missing_field():
    with pytest.raises(ValidationError):
        DownloadConfig(source="local")  # type: ignore[call-arg]


def test_config_rejects_extra_key():
    with pytest.raises(ValidationError):
        DownloadConfig(
            source="local",
            max_files_to_download=None,
            ignore_state=False,
            bogus=1,  # type: ignore[call-arg]
        )
