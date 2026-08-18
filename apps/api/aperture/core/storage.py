"""Object storage abstraction for raw episode blobs (video / point-cloud / raw logs).

Local default: filesystem under .aperture_data/blobs, with a signed-URL shim that returns
a file:// path. Production: Cloudflare R2 (S3-compatible, 10GB free, zero egress) — set the
R2_* env vars and the S3 backend activates with real presigned URLs. The interface is
identical so no calling code changes between the two.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from aperture.core.config import get_settings


class StorageBackend:
    def put_bytes(self, key: str, data: bytes) -> str:  # returns a uri
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def signed_url(self, key: str, expires_in: int = 3600) -> str:
        raise NotImplementedError


class LocalStorage(StorageBackend):
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_bytes(self, key: str, data: bytes) -> str:
        self._path(key).write_bytes(data)
        return f"local://{key}"

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def signed_url(self, key: str, expires_in: int = 3600) -> str:
        # Local demo: hand back a path the dev server can serve. Same shape as a presigned URL.
        return f"/v1/blobs/{key}"


class R2Storage(StorageBackend):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str) -> None:
        import boto3  # imported lazily so local runs don't require configured creds

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def put_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"r2://{self.bucket}/{key}"

    def get_bytes(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def signed_url(self, key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


_backend: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _backend
    if _backend is not None:
        return _backend
    s = get_settings()
    if s.r2_endpoint_url and s.r2_access_key_id and s.r2_secret_access_key:
        _backend = R2Storage(s.r2_endpoint_url, s.r2_access_key_id, s.r2_secret_access_key, s.r2_bucket)
    else:
        _backend = LocalStorage(s.local_blob_dir)
    return _backend


def reset_storage() -> None:
    """Test hook."""
    global _backend
    _backend = None


def wipe_local_blobs() -> None:
    s = get_settings()
    p = Path(s.local_blob_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
