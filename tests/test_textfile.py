"""agent.textfile: files keep their encoding and byte-order mark through an edit; unstorable text is refused, never mangled."""

from __future__ import annotations

import codecs

import pytest

from agent import textfile as tf


@pytest.mark.parametrize("raw, enc, bom", [
    ("Grüße € x\n".encode("utf-8"), "utf-8", b""),
    (codecs.BOM_UTF8 + "Grüße\n".encode("utf-8"), "utf-8", codecs.BOM_UTF8),
    ("Grüße € x\n".encode("cp1252"), "cp1252", b""),
    (b"caf\xe9 \x81\n", "latin-1", b""),                       # 0x81 is undefined in cp1252: latin-1 still round-trips
    (codecs.BOM_UTF16_LE + "hé\n".encode("utf-16-le"), "utf-16-le", codecs.BOM_UTF16_LE),
])
def test_decode_then_encode_gives_the_same_bytes(raw, enc, bom):
    text, info = tf.decode(raw)
    assert (info.encoding, info.bom) == (enc, bom)
    assert tf.encode(text, info) == raw


def test_unencodable_text_is_refused_with_a_clear_reason():
    with pytest.raises(tf.UnencodableText, match="cp1252.*U\\+2192"):
        tf.encode("a → b", tf.Info("cp1252"))


def test_write_atomic_keeps_the_existing_encoding_and_mode(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_bytes("Größe: 1\r\n".encode("cp1252"))
    f.chmod(0o640)
    text, _info = tf.read(f)
    tf.write_atomic(f, text.replace("1", "2"))
    assert f.read_bytes() == "Größe: 2\r\n".encode("cp1252")
    assert oct(f.stat().st_mode & 0o777) == "0o640"


def test_write_atomic_refuses_and_leaves_the_file_untouched(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_bytes("Größe\n".encode("cp1252"))
    with pytest.raises(tf.UnencodableText):
        tf.write_atomic(f, "Größe → neu\n")
    assert f.read_bytes() == "Größe\n".encode("cp1252") and len(list(tmp_path.iterdir())) == 1


def test_new_files_are_utf8_and_utf16_is_not_binary():
    assert tf.encode("ü", tf.UTF8) == "ü".encode()
    assert not tf.looks_binary(codecs.BOM_UTF16_LE + "abc".encode("utf-16-le"))
    assert tf.looks_binary(b"\x7fELF\0\0\0")


def test_chat_file_tools_keep_the_encoding_and_refuse_unstorable_text(tmp_path):
    from agent.tools import files

    f = tmp_path / "liste.txt"
    f.write_bytes("Äpfel\r\nBirnen\r\n".encode("cp1252"))
    assert files.replace_in_file(str(f), "Birnen", "Kirschen").startswith("Replaced")
    assert f.read_bytes() == "Äpfel\r\nKirschen\r\n".encode("cp1252")
    assert "cannot hold" in files.replace_in_file(str(f), "Kirschen", "Kirschen → rot")
    assert f.read_bytes() == "Äpfel\r\nKirschen\r\n".encode("cp1252")
    files.write_file(str(f), "Ölbaum\r\n", append=True)
    assert f.read_bytes() == "Äpfel\r\nKirschen\r\nÖlbaum\r\n".encode("cp1252")
