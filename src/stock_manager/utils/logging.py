from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback is exercised when tqdm is absent
    tqdm = None

ItemT = TypeVar("ItemT")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a simple default handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def progress_iter(
    iterable: Iterable[ItemT],
    *,
    total: int | None = None,
    desc: str,
    enabled: bool = True,
) -> Iterable[ItemT]:
    """Wrap an iterable with tqdm when available, otherwise emit periodic log updates."""
    if not enabled:
        return iterable

    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)

    logger = get_logger("stock_manager.progress")

    def _iterator() -> Iterator[ItemT]:
        if total is not None:
            logger.info("%s started (%s items)", desc, total)
            step = max(1, total // 10)
        else:
            logger.info("%s started", desc)
            step = 100

        count = 0
        for count, item in enumerate(iterable, start=1):
            if total is not None:
                if count == 1 or count == total or count % step == 0:
                    logger.info("%s %s/%s", desc, count, total)
            elif count == 1 or count % step == 0:
                logger.info("%s processed %s items", desc, count)
            yield item

        if total is not None:
            logger.info("%s completed (%s/%s)", desc, count, total)
        else:
            logger.info("%s completed (%s items)", desc, count)

    return _iterator()

