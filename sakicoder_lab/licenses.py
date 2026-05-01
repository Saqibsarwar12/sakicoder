from __future__ import annotations

from typing import Optional

ALLOWLIST = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0",
    "Unlicense",
    "MPL-2.0",
}

WARN_LIST = {
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-3.0",
    "LGPL",
}


def normalize_license(name: Optional[str]) -> str:
    if not name:
        return "UNKNOWN"
    normalized = name.strip().upper().replace(" ", "-")
    replacements = {
        "APACHE-2": "Apache-2.0",
        "APACHE-2.0": "Apache-2.0",
        "MIT": "MIT",
        "BSD-2-CLAUSE": "BSD-2-Clause",
        "BSD-3-CLAUSE": "BSD-3-Clause",
        "CC0": "CC0",
        "UNLICENSE": "Unlicense",
        "MPL-2.0": "MPL-2.0",
        "GPL-2.0": "GPL-2.0",
        "GPL-3.0": "GPL-3.0",
        "AGPL-3.0": "AGPL-3.0",
        "LGPL": "LGPL",
    }
    return replacements.get(normalized, normalized)


def is_license_allowed(name: Optional[str]) -> bool:
    normalized = normalize_license(name)
    return normalized in ALLOWLIST


def license_warning(name: Optional[str]) -> str:
    normalized = normalize_license(name)
    if normalized == "UNKNOWN":
        return "unknown license defaults to not allowed for training"
    if normalized in WARN_LIST:
        return f"{normalized} is strong copyleft; review carefully"
    if normalized not in ALLOWLIST:
        return f"{normalized} is not on the allowlist"
    return ""
