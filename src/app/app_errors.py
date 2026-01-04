"""
Module: Application Error Definitions.

Description:
    Defines a hierarchy of user-facing exceptions. These are caught by the UI layer
    to present clean, actionable feedback instead of raw stack traces.

    Design Philosophy:
    - User Message: What happened? (Non-technical)
    - Remediation: What can I do about it? (Actionable)
    - Details: Technical context for debugging (Hidden/Expandable)

Usage:
    Raise these in backend modules. Catch them in `streamlit_app.py`.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AppError(Exception):
    """
    Base class for all application-specific errors.
    
    Attributes:
        user_message (str): A friendly error summary.
        remediation (str): Steps the user can take to fix the issue.
        details (str): Technical details (e.g., raw API response).
    """
    user_message: str
    remediation: str = ""
    details: str = ""

    def __str__(self) -> str:
        parts = [self.user_message]
        if self.details:
            parts.append(f"Details: {self.details}")
        return " - ".join(parts)


class ConfigError(AppError):
    """
    Raised when required configuration (ENV vars, API keys) is missing or invalid.
    Example: Missing GOOGLE_MAPS_API_KEY.
    """


class ExternalServiceError(AppError):
    """
    Raised when an external API (Google Maps, Tankerkönig) fails or times out.
    Example: 503 Service Unavailable or Quota Exceeded.
    """


class DataAccessError(AppError):
    """
    Raised during database operations (Supabase) or file I/O.
    Example: Connection refused or SQL syntax error.
    """


class DataQualityError(AppError):
    """
    Raised when data is retrieved successfully but is semantically invalid.
    Example: A station has no price history or missing coordinates.
    """


class PredictionError(AppError):
    """
    Raised during the ML inference phase.
    Example: Model file missing or feature vector shape mismatch.
    """