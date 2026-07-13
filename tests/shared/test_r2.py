"""Tests for the R2 encryption + object-mapping core (src/pipeline/shared/r2.py).

No network: these exercise the self-contained blob format and the local<->key
mapping only.
"""

import struct

import pytest

from pipeline.shared import r2
from pipeline.shared.paths import DATA_DIR

PW = "correct horse battery staple"
AAD = b"raw/tok/photo.jpg"
PLAINTEXT = b"pretend these are jpeg bytes"


def test_encrypt_decrypt_roundtrip():
    blob = r2.encrypt(PLAINTEXT, PW, AAD)
    assert r2.decrypt(blob, PW, AAD) == PLAINTEXT


def test_blob_header_is_self_describing():
    # magic + version travel in the header so a blob decrypts from R2 alone.
    blob = r2.encrypt(PLAINTEXT, PW, AAD)
    magic, version = struct.unpack_from("<4sB", blob)
    assert magic == r2.MAGIC
    assert version == r2.FORMAT_VERSION


def test_each_blob_uses_fresh_salt_and_nonce():
    # Same inputs must not produce identical ciphertext (random salt + nonce).
    assert r2.encrypt(PLAINTEXT, PW, AAD) != r2.encrypt(PLAINTEXT, PW, AAD)


def test_wrong_passphrase_rejected():
    blob = r2.encrypt(PLAINTEXT, PW, AAD)
    with pytest.raises(Exception):
        r2.decrypt(blob, "wrong passphrase", AAD)


def test_wrong_aad_rejected():
    # AAD binds a blob to its logical path; a different path must fail auth.
    blob = r2.encrypt(PLAINTEXT, PW, AAD)
    with pytest.raises(Exception):
        r2.decrypt(blob, PW, b"raw/tok/other.jpg")


def test_tampered_ciphertext_rejected():
    blob = bytearray(r2.encrypt(PLAINTEXT, PW, AAD))
    blob[-1] ^= 0x01
    with pytest.raises(Exception):
        r2.decrypt(bytes(blob), PW, AAD)


def test_bad_magic_rejected_with_clear_error():
    with pytest.raises(ValueError, match="bad magic"):
        r2.decrypt(b"XXXX" + b"\x00" * 40, PW, AAD)


def test_r2_key_and_local_from_key_are_inverse():
    local = DATA_DIR / "raw" / "tok" / "photo.jpg"
    key = r2.r2_key(local)
    assert key == "raw/tok/photo.jpg" + r2.ENC_SUFFIX
    assert r2.local_from_key(key) == local


def test_curated_inputs_key_under_the_data_root():
    # Gramps genealogy lives under data/gramps, giving a "gramps/..." namespace.
    local = DATA_DIR / "gramps" / "portraits" / "erich.jpg"
    key = r2.r2_key(local)
    assert key == "gramps/portraits/erich.jpg" + r2.ENC_SUFFIX
    assert r2.local_from_key(key) == local


def test_aad_is_data_relative_posix():
    local = DATA_DIR / "raw" / "tok" / "photo.jpg"
    assert r2.aad_for(local) == b"raw/tok/photo.jpg"
