"""Nextcloud photo source (legacy).

Fetches files from Nextcloud public WebDAV shares into `data/raw/<token>/` and,
for the encrypted `.zip` archives the shares contain, extracts them in place
(a zip is Nextcloud packaging, not a pipeline concern). Retained only as a
historical ingest path; Nextcloud is being decommissioned in favour of R2 (see
docs/going-public.md).

Needs NEXTCLOUD_PASSWORD (share auth) and ZIP_PASSWORD (archive decryption) in
`.env`. State in data/state/download.json — a file is marked done once it has
been both downloaded and extracted.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict
from urllib.parse import unquote

import pyzipper
import requests
from loguru import logger

from pipeline.download.config import SharesConfig
from pipeline.download.sources.base import RAW_DIR
from pipeline.shared import state as state_lib

STATE_FILE = state_lib.STATE_DIR / "download.json"

DAV_NS = "DAV:"
PROPFIND_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:resourcetype/>
    <d:getcontentlength/>
    <d:displayname/>
  </d:prop>
</d:propfind>"""


class FileInfo(TypedDict):
    name: str
    href: str
    size: int


def _webdav_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/public.php/webdav/"


def _list_share_files(base_url: str, token: str, password: str) -> list[FileInfo]:
    response = requests.request(
        "PROPFIND",
        _webdav_url(base_url),
        auth=(token, password),
        headers={"Depth": "1", "Content-Type": "application/xml"},
        data=PROPFIND_BODY,
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    files: list[FileInfo] = []
    for resp in root.iter(f"{{{DAV_NS}}}response"):
        # Skip collections (<d:resourcetype><d:collection/></d:resourcetype>)
        resource_type = resp.find(f".//{{{DAV_NS}}}resourcetype")
        if (
            resource_type is not None
            and resource_type.find(f"{{{DAV_NS}}}collection") is not None
        ):
            continue

        href = resp.findtext(f"{{{DAV_NS}}}href", "")
        name = resp.findtext(f".//{{{DAV_NS}}}displayname", "") or unquote(
            href.rstrip("/").split("/")[-1]
        )
        size_text = resp.findtext(f".//{{{DAV_NS}}}getcontentlength", "0")
        files.append(FileInfo(name=name, href=href, size=int(size_text)))
    return files


def _download_file(base_url: str, token: str, password: str, href: str, dest: Path) -> None:
    # href from PROPFIND is the canonical path to the file
    url = f"{base_url.rstrip('/')}{href}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, auth=(token, password), stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def _extract_zip(zip_path: Path, password: str) -> list[str]:
    extracted: list[str] = []
    with pyzipper.AESZipFile(zip_path) as zf:
        zf.setpassword(password.encode())
        for member in zf.infolist():
            zf.extract(member, path=zip_path.parent)
            extracted.append(str(zip_path.parent / member.filename))
    return extracted


class NextcloudSource:
    def __init__(self, shares: SharesConfig) -> None:
        self._shares = shares

    def fetch(self, max_files: int | None, ignore_state: bool) -> None:
        nc_password = os.environ.get("NEXTCLOUD_PASSWORD", "")
        zip_password = os.environ.get("ZIP_PASSWORD", "")
        if not nc_password:
            logger.error("NEXTCLOUD_PASSWORD is not set in .env")
            return
        if not zip_password:
            logger.error("ZIP_PASSWORD is not set in .env")
            return

        base_url = str(self._shares.nextcloud_base_url).rstrip("/")
        state = state_lib.load(STATE_FILE)

        downloaded = skipped = failed = attempted = 0
        for share in self._shares.shares:
            if max_files is not None and attempted >= max_files:
                break
            token = share.token
            logger.info(f"Share [{token}] {share.description}")

            try:
                files = _list_share_files(base_url, token, nc_password)
            except Exception as e:
                logger.error(f"Failed to list share {token}: {e}")
                failed += 1
                attempted += 1
                continue
            logger.info(f"  {len(files)} file(s) found")

            for file in files:
                if max_files is not None and attempted >= max_files:
                    break
                name = file["name"]
                key = f"{token}/{name}"

                if not ignore_state and state_lib.is_done(state, key):
                    skipped += 1
                    continue

                attempted += 1
                dest = RAW_DIR / token / name
                logger.info(f"  Fetching {name} ({file['size'] / 1e6:.1f} MB)")
                try:
                    _download_file(base_url, token, nc_password, file["href"], dest)
                    # A .zip is Nextcloud packaging — extract it; anything else is
                    # already a raw file.
                    extracted = (
                        _extract_zip(dest, zip_password)
                        if dest.suffix.lower() == ".zip"
                        else []
                    )
                    state_lib.mark_done(state, key, [str(dest), *extracted])
                    state_lib.save(state, STATE_FILE)
                    detail = f"extracted {len(extracted)}" if extracted else "saved"
                    logger.success(f"  {name}: downloaded, {detail}")
                    downloaded += 1
                except Exception as e:
                    state_lib.mark_failed(state, key, str(e))
                    state_lib.save(state, STATE_FILE)
                    logger.error(f"  Failed {name}: {e}")
                    failed += 1

        logger.info(
            f"Done — nextcloud: {downloaded} downloaded+extracted, "
            f"{skipped} skipped (already done), {failed} failed"
        )
