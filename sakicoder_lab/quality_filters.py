from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import List

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"(?:sk-|rk-|ghp_|gho_|ghu_|github_pat_)\w+", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
]

SUSPICIOUS_COMMANDS = [
    re.compile(r"\bcurl\b.*\|\s*sh", re.IGNORECASE),
    re.compile(r"\bwget\b.*\|\s*sh", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
]

MINIFIED_HINT = re.compile(r"function\w+\(|;\w+=\w+|\{\w+:\w+\}")


@dataclass
class QualityResult:
    keep: bool
    issues: List[str]
    redacted_text: str


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    redacted = re.sub(r"(?:ssh-rsa|ssh-ed25519)\s+[A-Za-z0-9+/=]{20,}", "[REDACTED]", redacted)
    redacted = re.sub(r"(?i)password\s*[:=]\s*[^\s\n]+", "password=[REDACTED]", redacted)
    return redacted


def check_quality(text: str) -> dict:
    issues: List[str] = []
    raw = text or ""
    redacted = redact_secrets(raw)
    stripped = redacted.strip()

    if not stripped:
        issues.append("empty_content")
    if len(stripped) < 20:
        issues.append("too_short")
    if any(len(line) > 500 for line in stripped.splitlines() if line):
        issues.append("extremely_long_single_line")
    if _looks_binary(stripped):
        issues.append("binary_looking_content")
    if _looks_repeated(stripped):
        issues.append("repeated_junk")
    if _looks_minified(stripped):
        issues.append("minified_js_css")
    if any(pattern.search(stripped) for pattern in SECRET_PATTERNS):
        issues.append("secrets_or_tokens")
    if any(pattern.search(stripped) for pattern in SUSPICIOUS_COMMANDS):
        issues.append("suspicious_shell_command")

    keep = len(issues) == 0
    return {"keep": keep, "issues": issues, "redacted_text": redacted}


def should_keep_example(text: str) -> bool:
    return check_quality(text)["keep"]


def _looks_binary(text: str) -> bool:
    return "\x00" in text or sum(ord(ch) < 9 for ch in text) > max(len(text) // 3, 1)


def _looks_repeated(text: str) -> bool:
    lowered = text.lower().strip()
    return len(lowered) > 60 and len(set(lowered)) < max(10, len(lowered) // 10)


def _looks_minified(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(MINIFIED_HINT.search(compact)) and len(compact) > 250 and compact.count(";") > 8
