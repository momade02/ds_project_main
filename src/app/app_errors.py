"""
Centralized, user-facing exception types for the Streamlit app.

Design goals:
- Small set of meaningful categories.
- Actionable user guidance (remediation).
- Preserve technical context for debugging (details + chained cause).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppError(Exception):
    """Base class for errors that should be presented to end users."""

    user_message: str
    remediation: str = ""
    details: str = ""

    def __str__(self) -> str:  # pragma: no cover
        parts = [self.user_message]
        if self.details:
            parts.append(self.details)
        return " - ".join(parts)


class ConfigError(AppError):
    """Missing/invalid configuration (API keys, env vars, etc.)."""


class ExternalServiceError(AppError):
    """Failures calling external services (Google APIs, Tankerkönig, etc.)."""


class DataAccessError(AppError):
    """Failures reading/writing data stores (Supabase, CSV download, etc.)."""


class DataQualityError(AppError):
    """Data exists but is not usable (missing fields, inconsistent schema, etc.)."""


class PredictionError(AppError):
    """Model load / inference failures."""
