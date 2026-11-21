"""Единый логгер на loguru с fallback на stdlib logging."""

from __future__ import annotations

import sys

try:
    from loguru import logger as _logger

    _loguru_available = True
except ImportError:
    import logging

    _loguru_available = False


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Настроить единый формат логов. Идемпотентно.

    Заодно подгружает .env проекта: configure_logging — первый вызов каждого
    CLI-скрипта, до чтения путей к данным из окружения."""
    from deepfakes.utils.env import load_project_dotenv

    load_project_dotenv()
    global _CONFIGURED
    if _CONFIGURED:
        return
    if _loguru_available:
        _logger.remove()
        _logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
    else:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
            level=level,
        )
    _CONFIGURED = True


def get_logger(name: str = "deepfakes"):
    """Получить логгер с автоконфигурацией."""
    configure_logging()
    if _loguru_available:
        return _logger.bind(name=name)
    return logging.getLogger(name)
