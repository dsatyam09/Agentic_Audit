"""Load repo-root ``.env`` and register Hugging Face auth before any Hub/model calls.

Import this module first in any file that uses ``sentence_transformers``,
``transformers``, or other code that hits the Hugging Face Hub.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap() -> None:
    env_path = _REPO_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        # Env var alone is still enough for many code paths
        pass


def hub_auth_token() -> str | None:
    """Return HF token for authenticated downloads when configured.

    Returning ``None`` keeps Hub calls anonymous instead of forcing auth.
    """

    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return t if t else None


class _SuppressSpuriousHubUnauthWarning(logging.Filter):
    """Hub sometimes returns ``X-HF-Warning`` about unauthenticated requests even when
    ``HF_TOKEN`` is set (e.g. mixed anonymous + auth calls). If a token is configured,
    drop that specific advisory so logs stay readable.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not (
            os.environ.get("HF_TOKEN", "").strip()
            or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        ):
            return True
        msg = record.getMessage().lower()
        if "unauthenticated requests" in msg:
            return False
        return True


def _install_hf_hub_log_filter() -> None:
    logging.getLogger("huggingface_hub.utils._http").addFilter(_SuppressSpuriousHubUnauthWarning())


_bootstrap()
_install_hf_hub_log_filter()
