"""Platform adapters for Anima.

This module provides abstract interfaces for platform integration,
allowing Anima to work with different social platforms.
"""

from .protocol import PlatformAdapter, PlatformPost
from .threads import ThreadsAdapter

__all__ = ["PlatformAdapter", "PlatformPost", "ThreadsAdapter"]
