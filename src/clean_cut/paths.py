from __future__ import annotations

import os
import sys
from pathlib import Path


def app_data_dir() -> Path:
    configured = os.environ.get("PRODUCE_CLEAN_CUT_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "运行数据"
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "ProduceCleanCut"
