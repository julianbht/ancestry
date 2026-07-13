"""R2 photo source: pull raw photos from Cloudflare R2, the canonical store.

Used to hydrate `data/raw/` on a fresh machine. Shares the encryption/transport
core in `pipeline.shared.r2` with the backup tool (`scripts/sync_private.py`);
this source is just its raw-only, disk-idempotent read side. Needs the R2_* keys
in `.env`.
"""

from __future__ import annotations

from botocore.exceptions import ClientError
from loguru import logger

from pipeline.download.sources.base import RAW_DIR
from pipeline.shared import r2
from pipeline.shared.paths import rel


class R2Source:
    def fetch(self, max_files: int | None, ignore_state: bool) -> None:
        try:
            settings = r2.load_settings()
        except r2.MissingSecretError as e:
            logger.error(str(e))
            return
        client = r2.make_client(settings)

        prefix = RAW_DIR.name + "/"  # "raw/"
        keys = r2.list_keys(client, settings.bucket, prefix)
        logger.info(f"{len(keys)} raw object(s) in bucket {settings.bucket}")

        pulled = skipped = failed = 0
        for key in keys:
            if max_files is not None and pulled >= max_files:
                break
            local = r2.local_from_key(key)

            # Idempotent by disk presence — hydration doesn't need a state file.
            if not ignore_state and local.exists():
                skipped += 1
                continue

            try:
                blob = client.get_object(Bucket=settings.bucket, Key=key)["Body"].read()
                data = r2.decrypt(blob, settings.passphrase, r2.aad_for(local))
            except (ClientError, ValueError) as e:
                logger.error(f"FAILED {key}: {e}")
                failed += 1
                continue

            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            pulled += 1
            logger.info(f"pulled {rel(local)} ({len(data) / 1e6:.1f} MB)")

        logger.info(
            f"Done — r2: {pulled} pulled, {skipped} skipped (already local), {failed} failed"
        )
