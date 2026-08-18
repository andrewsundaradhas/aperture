"""Application configuration.

Free-tier-first: everything defaults to a zero-cost local setup (SQLite + local
filesystem object store + a static demo API key). Production free-tier services drop in
by setting the corresponding environment variables:

  DATABASE_URL      -> Supabase Postgres connection string (postgresql+psycopg2://...)
  R2_*              -> Cloudflare R2 (S3-compatible) credentials + bucket
  APERTURE_API_KEYS -> comma-separated "key:org_slug" pairs for programmatic upload auth
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository-local data dir used by the SQLite db and the local object store.
_DATA_DIR = Path(__file__).resolve().parents[2] / ".aperture_data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APERTURE_", env_file=".env", extra="ignore")

    app_name: str = "Aperture API"
    environment: str = "local"

    # --- Database -------------------------------------------------------------
    # Default: file-backed SQLite. Set APERTURE_DATABASE_URL to a Supabase Postgres
    # URL for production.
    database_url: str = f"sqlite:///{(_DATA_DIR / 'aperture.db').as_posix()}"

    # --- Object storage (video/point-cloud blobs) -----------------------------
    # Default: local filesystem under .aperture_data/blobs. Set the R2_* values to use
    # Cloudflare R2 (S3-compatible, 10GB free, zero egress).
    r2_endpoint_url: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket: str = "aperture-blobs"
    local_blob_dir: str = str(_DATA_DIR / "blobs")

    # --- Auth -----------------------------------------------------------------
    # Comma-separated "key:org_slug" pairs. The default demo key maps to the seeded
    # "acme-robotics" org so the stack is usable out of the box. In production this is
    # backed by Supabase Auth + per-org API keys.
    api_keys: str = "demo-key:acme-robotics"

    # --- Upload safety limits (Phase 6) ---------------------------------------
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB per file
    max_batch_files: int = 50

    @property
    def data_dir(self) -> Path:
        return _DATA_DIR

    def api_key_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            key, org = pair.split(":", 1)
            out[key.strip()] = org.strip()
        return out


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.local_blob_dir).mkdir(parents=True, exist_ok=True)
    return settings
