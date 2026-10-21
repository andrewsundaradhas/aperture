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


class InvalidKey(ValueError):
    """A storage key that is not a plain relative path: absolute, `..`, backslashes, empty parts."""


def validate_key(key: str) -> str:
    """Reject any key that could resolve outside its own prefix.

    Keys reach storage from request paths (`/v1/blobs/{key}`), so a key is untrusted input. On the
    filesystem backend `acme/../../etc/passwd` would escape the blob root; on S3 the same string is
    a literal object name, but rejecting it everywhere keeps one rule for both backends.
    """
    if not key or "\\" in key or "\x00" in key or key.startswith("/"):
        raise InvalidKey(f"invalid storage key: {key!r}")
    parts = key.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise InvalidKey(f"invalid storage key: {key!r}")
    return key


def key_belongs_to(key: str, prefix: str) -> bool:
    """True when `key`'s first path segment is exactly `prefix` (an org slug)."""
    try:
        validate_key(key)
    except InvalidKey:
        return False
    return key.split("/", 1)[0] == prefix and "/" in key


class StorageBackend:
    def put_bytes(self, key: str, data: bytes) -> str:  # returns a uri
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def put_fileobj(self, key: str, fileobj) -> str:
        """Store a file-like object without holding it in memory. Returns the uri."""
        return self.put_bytes(key, fileobj.read())

    def signed_url(self, key: str, expires_in: int = 3600) -> str:
        raise NotImplementedError

    def signed_upload_url(self, key: str, expires_in: int = 3600) -> str:
        """A URL the client can PUT bytes to directly, bypassing the API process.

        A LeRobot v3 dataset of real fleet video is gigabytes. Proxying that through the API
        means holding it in memory to hand to the parser, and paying for the bandwidth twice.
        The client uploads straight to object storage and then tells the API the key.
        """
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def size(self, key: str) -> int:
        """Stored object size in bytes, without reading it."""
        raise NotImplementedError


class LocalStorage(StorageBackend):
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        validate_key(key)
        root = self.root.resolve()
        p = (root / key).resolve()
        # Belt and braces after validate_key: a symlink inside the root must not lead out of it.
        if not p.is_relative_to(root):
            raise InvalidKey(f"storage key escapes the blob root: {key!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_bytes(self, key: str, data: bytes) -> str:
        self._path(key).write_bytes(data)
        return f"local://{key}"

    def put_fileobj(self, key: str, fileobj) -> str:
        with open(self._path(key), "wb") as out:
            shutil.copyfileobj(fileobj, out, 1024 * 1024)
        return f"local://{key}"

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def signed_url(self, key: str, expires_in: int = 3600) -> str:
        # Local demo: hand back a path the dev server can serve. Same shape as a presigned URL.
        return f"/v1/blobs/{key}"

    def signed_upload_url(self, key: str, expires_in: int = 3600) -> str:
        """Local equivalent: the API's own PUT route. Same client flow as a real presigned URL,
        so nothing downstream has to know which backend is in use."""
        return f"/v1/blobs/{key}"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size


class R2Storage(StorageBackend):
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        public_endpoint: str | None = None,
    ) -> None:
        import boto3  # imported lazily so local runs don't require configured creds

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        # Presigned URLs are opened by the *client's* browser or uploader, and a SigV4 signature
        # covers the host — so they must be signed for an address the client can reach. Inside
        # docker-compose the API talks to `http://storage:9000`, a name that only resolves on
        # the compose network; signing for it handed every uploader a dead link.
        self.sign_client = (
            boto3.client(
                "s3",
                endpoint_url=public_endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            if public_endpoint and public_endpoint != endpoint
            else self.client
        )

    def put_bytes(self, key: str, data: bytes) -> str:
        validate_key(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"r2://{self.bucket}/{key}"

    def put_fileobj(self, key: str, fileobj) -> str:
        validate_key(key)
        self.client.upload_fileobj(fileobj, self.bucket, key)  # multipart for large files
        return f"r2://{self.bucket}/{key}"

    def get_bytes(self, key: str) -> bytes:
        validate_key(key)
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def signed_url(self, key: str, expires_in: int = 3600) -> str:
        return self.sign_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def signed_upload_url(self, key: str, expires_in: int = 3600) -> str:
        validate_key(key)
        return self.sign_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def size(self, key: str) -> int:
        validate_key(key)
        return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])


_backend: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _backend
    if _backend is not None:
        return _backend
    s = get_settings()
    if s.r2_endpoint_url and s.r2_access_key_id and s.r2_secret_access_key:
        _backend = R2Storage(
            s.r2_endpoint_url, s.r2_access_key_id, s.r2_secret_access_key, s.r2_bucket,
            public_endpoint=s.r2_public_endpoint_url,
        )
    else:
        _backend = LocalStorage(s.local_blob_dir)
    return _backend


def key_from_uri(uri: str) -> str:
    """Recover the storage key from a uri returned by `put_bytes` (local:// or r2://bucket/)."""
    if uri.startswith("local://"):
        return uri[len("local://") :]
    if uri.startswith("r2://"):
        rest = uri[len("r2://") :]
        return rest.split("/", 1)[1] if "/" in rest else rest
    return uri


def read_uri(uri: str) -> bytes:
    """Read the bytes a stored uri points at, regardless of backend."""
    return get_storage().get_bytes(key_from_uri(uri))


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
