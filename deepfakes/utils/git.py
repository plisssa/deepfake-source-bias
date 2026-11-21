"""Получение git-метаданных для воспроизводимости."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_info(repo_root: Path | str | None = None) -> dict[str, str]:
    """Получить git commit, branch, dirty-flag."""
    cwd = Path(repo_root) if repo_root else None
    commit = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    status = _run(["git", "status", "--porcelain"], cwd=cwd)
    return {
        "git_commit": commit[:12] if commit != "unknown" else commit,
        "git_branch": branch,
        "git_dirty": "true" if status and status != "unknown" else "false",
    }
