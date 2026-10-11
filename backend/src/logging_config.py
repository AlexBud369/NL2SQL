import logging.config

from src.config import settings


def setup_logging() -> None:
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "filename": "app.log",
                "encoding": "utf-8",
                "formatter": "default",
                "level": "INFO",
            },
            "console_handler": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "DEBUG",
            },
        },
        "loggers": {
            "src": {
                "handlers": ["file_handler", "console_handler"],
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["file_handler", "console_handler"],
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["file_handler", "console_handler"],
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["file_handler", "console_handler"],
                "level": settings.log_level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["file_handler", "console_handler"],
            "level": settings.log_level,
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)
