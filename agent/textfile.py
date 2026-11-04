"""Read and write text files without damaging them: the encoding and byte-order mark a file had are kept.

The coding agent and the chat file tools used to decode with `errors="replace"` and write UTF-8, so ONE edit of a Latin-1 or
Windows-1252 file turned every umlaut into U+FFFD - and the undo snapshot already held the damaged text.

  decode(raw)          -> (text, Info). A BOM decides first (UTF-8 / UTF-16 / UTF-32); else strict UTF-8; else cp1252 (Windows
                          "ANSI": umlauts, the euro sign); else latin-1, where every byte is one character, so the bytes always
                          round-trip.
  encode(text, info)   -> bytes in the same encoding (BOM included). Raises UnencodableText when the text holds a character that
                          encoding cannot store - the caller then writes nothing instead of writing something wrong.
  write_atomic(p, text, info=None)
                       -> temp file in the same folder + rename (a crash leaves the old file or the new one, never half of it), the
                          old file's mode kept. Without `info` an existing file keeps its own encoding; a new file is UTF-8.
  looks_binary(raw)    -> True for data with NUL bytes that is not UTF-16/32 text.

Line endings are the caller's business (the text keeps whatever `\\r\\n` it has: decode/encode never translate newlines)."""

from __future__ import annotations

import codecs
import contextlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Info", "UTF8", "UnencodableText", "decode", "encode", "read", "write_atomic", "looks_binary"]


@dataclass(frozen=True)
class Info:
    encoding: str = "utf-8"
    bom: bytes = b""

    @property
    def label(self) -> str:
        """A short human name: '' for plain UTF-8, else e.g. 'cp1252' or 'utf-8 with BOM'."""
        if self.encoding == "utf-8" and not self.bom:
            return ""
        return f"{self.encoding} with BOM" if self.bom else self.encoding


UTF8 = Info()


class UnencodableText(ValueError):
    """The new text holds a character the file's encoding cannot store."""


# UTF-32 first: its little-endian BOM begins with the UTF-16 one.
_BOMS = ((codecs.BOM_UTF32_LE, "utf-32-le"), (codecs.BOM_UTF32_BE, "utf-32-be"), (codecs.BOM_UTF8, "utf-8"),
         (codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be"))


def _bom_of(raw: bytes) -> tuple[bytes, str] | None:
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return bom, enc
    return None


def looks_binary(raw: bytes) -> bool:
    """NUL bytes in the first 4 KiB, unless a UTF-16/32 byte-order mark says it is text."""
    head = raw[:4096]
    if b"\0" not in head:
        return False
    found = _bom_of(raw)
    return not (found and found[1].startswith(("utf-16", "utf-32")))


def decode(raw: bytes) -> tuple[str, Info]:
    """The text of `raw` and how it was stored (see the module docstring). Never raises."""
    found = _bom_of(raw)
    if found:
        bom, enc = found
        try:
            return raw[len(bom):].decode(enc), Info(enc, bom)
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode("utf-8"), UTF8
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), Info("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1"), Info("latin-1")


def encode(text: str, info: Info = UTF8) -> bytes:
    try:
        data = text.encode(info.encoding)
    except UnicodeEncodeError as exc:
        ch = exc.object[exc.start]
        raise UnencodableText(f"the file is stored as {info.encoding}, which cannot hold {ch!r} (U+{ord(ch):04X}); nothing was "
                              "written - use characters that encoding has (or plain ASCII)") from None
    return info.bom + data


def read(p: Path | str) -> tuple[str, Info]:
    return decode(Path(p).read_bytes())


def write_atomic(p: Path | str, text: str, info: Info | None = None) -> Info:
    """Write `text` to `p` atomically in `info`'s encoding (default: the encoding the existing file has, UTF-8 for a new one).
    Returns the Info used. Raises UnencodableText (nothing written) or OSError."""
    p = Path(p)
    if info is None:
        try:
            info = decode(p.read_bytes())[1] if p.is_file() else UTF8
        except OSError:
            info = UTF8
    data = encode(text, info)                       # before anything touches the disk
    mode = stat.S_IMODE(p.stat().st_mode) if p.exists() else None
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return info
