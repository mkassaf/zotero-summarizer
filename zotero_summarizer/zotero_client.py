"""Thin wrapper around pyzotero for the operations this app needs:
list a collection's papers, find/download their PDF, and write summary notes.

Works against either the local Zotero desktop API (ZOTERO_LOCAL=true) or the
Zotero Web API. Both are exposed through the same pyzotero client object."""

from __future__ import annotations

import os
import re
from typing import List, Optional

import requests  # pulled in by pyzotero
from pyzotero import zotero

from .config import Settings


def _resolve_user_id(library_id: str, api_key: str) -> str:
    """The Zotero Web API needs the numeric userID. If a username was given
    instead, look up the numeric id via the API key's account."""
    if library_id.isdigit():
        return library_id
    resp = requests.get(
        "https://api.zotero.org/keys/current",
        headers={"Zotero-API-Key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    info = resp.json()
    numeric = str(info.get("userID"))
    username = info.get("username", "")
    if username and library_id.strip().lower() != username.lower():
        raise ValueError(
            f"ZOTERO_LIBRARY_ID='{library_id}' doesn't match this API key's "
            f"account (username '{username}', id {numeric}). Use your own "
            f"username, your numeric id, or a key for that account."
        )
    return numeric

# Substring embedded in every note we create, so re-runs can detect and skip
# papers that already have an AI summary.
SUMMARY_MARKER = "zotero-summarizer"

# Zotero item/collection keys are 8 uppercase alphanumerics.
_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")

# Item types that are never "papers".
_NON_PAPER_TYPES = {"attachment", "note", "annotation"}


class ZoteroClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.zotero_local:
            self.zot = zotero.Zotero(
                settings.zotero_library_id,
                settings.zotero_library_type,
                local=True,
            )
        else:
            library_id = settings.zotero_library_id
            # Accept a username for user libraries and resolve it to the
            # numeric id the Web API requires.
            if settings.zotero_library_type == "user" and settings.zotero_api_key:
                library_id = _resolve_user_id(library_id, settings.zotero_api_key)
            self.zot = zotero.Zotero(
                library_id,
                settings.zotero_library_type,
                settings.zotero_api_key,
            )

    # -- collections -----------------------------------------------------
    def resolve_collection_key(self, name_or_key: str) -> str:
        """Accept either an 8-char collection key or a (case-insensitive) name."""
        if _KEY_RE.match(name_or_key):
            return name_or_key
        for col in self.zot.everything(self.zot.collections()):
            if col["data"]["name"].lower() == name_or_key.lower():
                return col["key"]
        raise ValueError(
            f"No collection named '{name_or_key}' found. "
            f"Pass the exact name or its 8-character key."
        )

    def papers_in_collection(self, name_or_key: str) -> List[dict]:
        """Top-level items in the collection, excluding notes/attachments."""
        key = self.resolve_collection_key(name_or_key)
        items = self.zot.everything(self.zot.collection_items_top(key))
        return [
            it for it in items
            if it["data"].get("itemType") not in _NON_PAPER_TYPES
        ]

    # -- attachments -----------------------------------------------------
    def find_pdf_attachment(self, item_key: str) -> Optional[dict]:
        for child in self.zot.children(item_key):
            data = child["data"]
            if data.get("itemType") != "attachment":
                continue
            content_type = data.get("contentType", "")
            filename = data.get("filename", "") or ""
            if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
                return child
        return None

    def download_pdf(self, attachment: dict) -> Optional[bytes]:
        """Fetch PDF bytes via the API, falling back to the local storage dir."""
        key = attachment["key"]

        # 1) API file endpoint (works for Web API w/ storage, and local API).
        try:
            data = self.zot.file(key)
            if data:
                return data
        except Exception:
            pass

        # 2) Read directly from the Zotero storage folder if configured.
        storage = self.settings.zotero_storage_dir
        if storage:
            folder = os.path.join(storage, key)
            filename = attachment["data"].get("filename")
            if filename:
                candidate = os.path.join(folder, filename)
                if os.path.exists(candidate):
                    with open(candidate, "rb") as fh:
                        return fh.read()
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    if name.lower().endswith(".pdf"):
                        with open(os.path.join(folder, name), "rb") as fh:
                            return fh.read()
        return None

    # -- notes -----------------------------------------------------------
    def summary_note_exists(self, item_key: str) -> bool:
        for child in self.zot.children(item_key):
            data = child["data"]
            if data.get("itemType") == "note" and SUMMARY_MARKER in (data.get("note") or ""):
                return True
        return False

    def add_note(self, item_key: str, html: str) -> dict:
        # Build the note item locally rather than via GET /items/new, which the
        # local Zotero API does not implement.
        note = {
            "itemType": "note",
            "note": html,
            "parentItem": item_key,
            "collections": [],
            "tags": [],
            "relations": {},
        }
        return self.zot.create_items([note])
