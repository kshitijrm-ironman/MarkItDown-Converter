"""
Password-protected documents: detect them, and unlock them when the user
supplies the password.

Encrypted files are unreadable by MarkItDown / Unlimited-OCR / Tesseract, so
this module is called first: it decrypts the *temporary* copy of the upload in
place (the original upload is never touched) and the normal pipeline runs on
the plain file. When no password — or a wrong one — is given it raises
`PasswordRequired`, which the UI turns into a "this file needs a password"
message instead of a generic failure.

Supported: PDF (PyMuPDF) and MS Office / OLE containers (msoffcrypto-tool).
Encrypted ZIP members are handled by `file_handlers.convert_archive`.
"""
from __future__ import annotations

import os

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"   # .doc/.xls and encrypted OOXML


class PasswordRequired(Exception):
    """A file is encrypted and no password (or a wrong one) was supplied."""

    def __init__(self, name: str, wrong: bool = False):
        self.name = name
        self.wrong = wrong
        reason = "the password is incorrect" if wrong else "a password is required"
        super().__init__(f"'{name}' is password-protected — {reason}.")


def _head(path: str, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def unlock_in_place(path: str, name: str, password: str = "") -> bool:
    """
    Decrypt `path` in place when it is an encrypted PDF or Office file.

    Returns True when the file was encrypted and has been decrypted, False when
    it was not encrypted at all. Raises `PasswordRequired` when the password is
    missing or wrong.
    """
    try:
        head = _head(path)
    except OSError:
        return False

    if head.startswith(b"%PDF"):
        return _unlock_pdf(path, name, password)
    if head.startswith(_OLE_MAGIC):
        return _unlock_office(path, name, password)
    return False


def _unlock_pdf(path: str, name: str, password: str) -> bool:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        if not doc.needs_pass:
            return False
        if not password:
            raise PasswordRequired(name)
        if not doc.authenticate(password):
            raise PasswordRequired(name, wrong=True)
        tmp = path + ".unlocked.pdf"
        doc.save(tmp)            # saved without encryption
    finally:
        doc.close()
    os.replace(tmp, path)
    return True


def _unlock_office(path: str, name: str, password: str) -> bool:
    try:
        import msoffcrypto
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "This Office file looks encrypted. Install the unlocker first: "
            "`pip install msoffcrypto-tool`."
        ) from exc

    tmp = path + ".unlocked"
    with open(path, "rb") as f:
        try:
            office = msoffcrypto.OfficeFile(f)
            encrypted = office.is_encrypted()
        except Exception:  # noqa: BLE001 — not an OLE file msoffcrypto understands
            return False
        if not encrypted:
            return False
        if not password:
            raise PasswordRequired(name)
        try:
            office.load_key(password=password)
            with open(tmp, "wb") as out:
                office.decrypt(out)
        except Exception as exc:  # noqa: BLE001 — bad password or broken container
            if os.path.exists(tmp):
                os.remove(tmp)
            raise PasswordRequired(name, wrong=True) from exc
    os.replace(tmp, path)
    return True
