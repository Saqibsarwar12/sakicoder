from __future__ import annotations

import re

DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/s\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
]

NETWORK_DOWNLOAD_PATTERNS = [
    re.compile(r"\bcurl\b.*\|\s*sh", re.IGNORECASE),
    re.compile(r"\bwget\b.*\|\s*sh", re.IGNORECASE),
    re.compile(r"\bcurl\b.*https?://", re.IGNORECASE),
    re.compile(r"\bwget\b.*https?://", re.IGNORECASE),
]

RISKY_PATTERNS = [
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"chmod\s+777", re.IGNORECASE),
    re.compile(r"~/.ssh"),
    re.compile(r"\benv\b|printenv|set\s*$", re.IGNORECASE),
]


def is_destructive_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in DESTRUCTIVE_PATTERNS)


def is_network_download_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in NETWORK_DOWNLOAD_PATTERNS)


def requires_confirmation(command: str) -> bool:
    return is_destructive_command(command) or is_network_download_command(command) or any(pattern.search(command) for pattern in RISKY_PATTERNS)
