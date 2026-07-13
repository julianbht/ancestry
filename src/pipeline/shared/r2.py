"""Cloudflare R2 transport + at-rest encryption for private data.

R2 is the canonical, durable store for the project's private data (the family
photos and their derived/curated artifacts). Local disk is a working *cache* of
it — blow it away and reconstruct with a pull. See docs/going-public.md.

This module is the reusable core: passphrase-based AES-256-GCM encryption, the
boto3 client for R2, and the mapping between a local file and its R2 object key.
The push/pull orchestration + state tracking lives in scripts/sync_private.py,
which is a thin CLI over these primitives; anything else that needs to read or
write R2 (e.g. a future raw-photo hydrate path) imports from here rather than
reimplementing the transport.

Secrets come from ``.env`` (gitignored) — see .env.example for the key list.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from pipeline.shared.paths import DATA_DIR

# --- Encrypted blob format --------------------------------------------------
# Each blob is self-contained: it carries its own salt and nonce, so any object
# can be decrypted with the passphrase alone — a pull needs nothing but R2 and
# .env, which is what makes "reconstruct a machine from R2 alone" work.
MAGIC = b"AREN"  # Ancestry R2 ENcryption
FORMAT_VERSION = 1
SALT_LEN = 16
NONCE_LEN = 12
_HEADER = struct.Struct(f"<4sB{SALT_LEN}s{NONCE_LEN}s")  # magic, version, salt, nonce
ENC_SUFFIX = ".enc"

# scrypt cost params (interactive-strength; ~10-20 ms/derivation). N must be a
# power of two; r=8, p=1 are the standard companions.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LEN = 32  # AES-256


class MissingSecretError(RuntimeError):
    """A required R2 secret is absent from the environment / .env."""


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------
def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=_KEY_LEN, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt(plaintext: bytes, passphrase: str, aad: bytes) -> bytes:
    """Return a self-contained encrypted blob (header carries salt + nonce)."""
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = _derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    return _HEADER.pack(MAGIC, FORMAT_VERSION, salt, nonce) + ciphertext


def decrypt(blob: bytes, passphrase: str, aad: bytes) -> bytes:
    """Reverse of encrypt(). Raises on tampering, wrong passphrase, or bad AAD."""
    magic, version, salt, nonce = _HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("not an ancestry encrypted blob (bad magic)")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported blob format version {version}")
    key = _derive_key(passphrase, salt)
    ciphertext = blob[_HEADER.size :]
    return AESGCM(key).decrypt(nonce, ciphertext, aad)


# ---------------------------------------------------------------------------
# R2 client + config
# ---------------------------------------------------------------------------
@dataclass
class R2Settings:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    passphrase: str


def load_settings() -> R2Settings:
    """Read R2 settings from the environment (.env). Raises MissingSecretError
    naming the first missing key."""

    def req(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise MissingSecretError(
                f"Missing {name} in .env. Required keys: R2_ACCOUNT_ID, "
                "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PASSPHRASE."
            )
        return value

    return R2Settings(
        account_id=req("R2_ACCOUNT_ID"),
        access_key_id=req("R2_ACCESS_KEY_ID"),
        secret_access_key=req("R2_SECRET_ACCESS_KEY"),
        bucket=req("R2_BUCKET"),
        passphrase=req("R2_PASSPHRASE"),
    )


def make_client(s: R2Settings):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{s.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=s.access_key_id,
        aws_secret_access_key=s.secret_access_key,
        region_name="auto",  # R2 ignores region but boto3 wants one set
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    """All object keys under a prefix, sorted."""
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return sorted(keys)


# ---------------------------------------------------------------------------
# Local file <-> R2 object mapping
# ---------------------------------------------------------------------------
# The R2 key equals the file's path relative to the data root, plus .enc — one
# flat, human-readable namespace on R2 that mirrors the local layout (raw photos
# → "raw/...", curated inputs → "curated/..."). All private data lives under
# DATA_DIR, so a single relative-to root suffices.
def r2_key(local: Path) -> str:
    """R2 object key for a local file: its DATA_DIR-relative path plus .enc."""
    return local.relative_to(DATA_DIR).as_posix() + ENC_SUFFIX


def local_from_key(key: str) -> Path:
    """Inverse of r2_key(): the local path an R2 object restores to."""
    return DATA_DIR / key[: -len(ENC_SUFFIX)]


def aad_for(local: Path) -> bytes:
    """Associated data binding a blob to its logical location (its DATA_DIR-relative
    path), so a blob can't be silently relocated to a different key."""
    return local.relative_to(DATA_DIR).as_posix().encode("utf-8")
