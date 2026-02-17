from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging


def setup_logging(run_name: str, logs_dir: Path | str = "logs") -> Path:
    """
    Configure logging:
      - Console: INFO (readable)
      - File:    DEBUG (full details)
    Returns the log file path.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_path / f"{run_name}_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear previous handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root.addHandler(stream)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging to %s", log_file)
    return log_file
