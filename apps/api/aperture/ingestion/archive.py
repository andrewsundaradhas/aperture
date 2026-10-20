"""Safe unpacking of uploaded dataset archives.

A real dataset is a directory — LeRobot v3 is parquet tables plus mp4s plus a meta folder, and
an RLDS export is a set of `.tfrecord` shards — so it arrives over HTTP as one archive. Both
formats share this unpacker so the path-traversal defences below exist in exactly one place.

Uploads are untrusted input. A `.zip` or `.tar` member may name `../../etc/authorized_keys`,
or be a symlink pointing anywhere on the host; either would let an upload write outside the
extraction directory. Both are rejected rather than sanitised, because a dataset archive has
no legitimate reason to contain them.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


class ArchiveError(ValueError):
    """Raised when an archive is unreadable or tries to escape the extraction root."""


def is_archive(filename: str) -> bool:
    return filename.lower().endswith(ARCHIVE_SUFFIXES)


def _safe_target(dest: Path, name: str) -> Path:
    target = dest / name
    try:
        target.resolve().relative_to(dest.resolve())
    except ValueError as e:
        raise ArchiveError(f"archive member escapes the extraction root: {name!r}") from e
    return target


_CHUNK = 1 << 20


class _Budget:
    """Caps what one archive may expand to, counted from bytes actually written.

    Header-declared sizes are attacker-controlled (a zip can claim a 1 KB member and inflate to
    terabytes), so the count is of real output, checked chunk by chunk, and extraction stops the
    moment either limit is crossed — before the disk fills, not after.
    """

    def __init__(self, max_bytes: int, max_members: int) -> None:
        self.max_bytes, self.max_members = max_bytes, max_members
        self.bytes = 0
        self.members = 0

    def member(self, name: str) -> None:
        self.members += 1
        if self.members > self.max_members:
            raise ArchiveError(
                f"archive has more than {self.max_members} files; raise "
                "APERTURE_MAX_ARCHIVE_MEMBERS if this dataset is legitimate."
            )

    def copy(self, source, target: Path) -> None:
        with open(target, "wb") as out:
            while chunk := source.read(_CHUNK):
                self.bytes += len(chunk)
                if self.bytes > self.max_bytes:
                    raise ArchiveError(
                        f"archive expands past {self.max_bytes} bytes; raise "
                        "APERTURE_MAX_EXTRACTED_BYTES if this dataset is legitimate."
                    )
                out.write(chunk)


def unpack(raw: bytes, filename: str, dest: Path) -> Path:
    """Extract `raw` into `dest`, returning `dest`.

    Rejects traversal and link members, and stops once the expanded size or file count passes
    the configured limits (zip-bomb protection).
    """
    from aperture.core.config import get_settings

    settings = get_settings()
    budget = _Budget(settings.max_extracted_bytes, settings.max_archive_members)
    dest.mkdir(parents=True, exist_ok=True)
    lower = filename.lower()

    if lower.endswith(".zip"):
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as e:
            raise ArchiveError(f"'{filename}' is not a readable zip archive: {e}") from e
        with archive as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                budget.member(member.filename)
                target = _safe_target(dest, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source:
                    budget.copy(source, target)

    elif lower.endswith((".tar", ".tar.gz", ".tgz")):
        try:
            archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
        except tarfile.TarError as e:
            raise ArchiveError(f"'{filename}' is not a readable tar archive: {e}") from e
        with archive as tf:
            for member in tf.getmembers():
                if member.issym() or member.islnk():
                    raise ArchiveError(f"archive contains a link member: {member.name!r}")
                if not member.isfile():
                    continue
                budget.member(member.name)
                target = _safe_target(dest, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                if source is not None:
                    budget.copy(source, target)
    else:
        raise ArchiveError(
            f"'{filename}' is not a supported archive ({', '.join(ARCHIVE_SUFFIXES)})."
        )

    return dest
