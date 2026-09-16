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

# Where `ml/training/train.py` writes policy.pt / failure_head.pt in a source checkout. Absent
# when the package is installed standalone, in which case weights come from the Hub instead.
_REPO_MODEL_DIR = Path(__file__).resolve().parents[4] / "ml" / "models"


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
    # Frames kept per ingested episode. 0 means "keep every frame", which is the default:
    # truncating a trajectory silently teaches a fine-tune that the task ends early. Set a
    # positive cap only to bound memory on very long episodes — each truncation is logged.
    max_frames_per_episode: int = 0

    # --- Rate limiting --------------------------------------------------------
    # Per-organization token buckets (see aperture/core/ratelimit.py). Off in tests, which
    # deliberately hammer the API; on by default everywhere else.
    rate_limit_enabled: bool = True

    # --- Background jobs ------------------------------------------------------
    # Run the job worker on a thread inside the API process. Convenient for a single-machine
    # run (and what makes the README quick-start work); production runs
    # `python -m aperture.jobs.worker` as its own process and sets this false.
    inline_worker: bool = True

    # --- Learned models (optional; requires the `[ml]` extra) ------------------
    # Off by default so the base install runs anywhere with zero ML dependencies. When True
    # AND the `[ml]` extra is installed AND the checkpoints load, the classify / attribution /
    # clustering paths use the trained AperturePolicy + FailureHead over episode frame images,
    # falling back to the heuristic/simulated path per-episode when an image is absent.
    use_learned_models: bool = False
    model_device: str = "cpu"  # "cuda"/"mps" if the API host has a GPU
    # Fallback source when the checkpoints are not on disk. There is no public reference repo —
    # point this at one you own after publishing your own run (see ml/notebooks/train_aperture.ipynb).
    hf_model_repo: str = "KavinandHobbes/aperture-reference-policy"
    hf_policy_file: str = "policy.pt"
    hf_failure_head_file: str = "failure_head.pt"
    # Directory holding policy.pt / failure_head.pt. Defaults to the checkout's ml/models, which
    # is exactly where training writes them, so a local train makes the learned path work with no
    # further configuration. Falls back to the Hub when a file is missing here.
    local_model_dir: str | None = str(_REPO_MODEL_DIR) if _REPO_MODEL_DIR.is_dir() else None

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
