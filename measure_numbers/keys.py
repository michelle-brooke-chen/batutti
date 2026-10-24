"""API key lookup, following the course rule."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent


def portkey_key():
    """PORTKEY_API_KEY from .env in the repo folder, else one folder above.

    Both files are read, the repo's first, so a value there wins. Returns None
    if there is no key, so the app can still number parts without the agent.
    """
    for folder in (REPO, REPO.parent):
        if (folder / ".env").exists():
            load_dotenv(folder / ".env")  # never overrides a value already set
    return os.getenv("PORTKEY_API_KEY") or None
