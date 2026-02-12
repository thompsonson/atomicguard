"""Backwards-compatible re-exports.

StagnationInfo and FeedbackSummarizer have moved to the domain layer
where they belong (Issue 1: they are pure domain concepts).

Import from atomicguard.domain instead:
    from atomicguard.domain.models import StagnationInfo
    from atomicguard.domain.feedback_summarizer import FeedbackSummarizer
"""

from atomicguard.domain.feedback_summarizer import FeedbackSummarizer
from atomicguard.domain.models import StagnationInfo

__all__ = ["FeedbackSummarizer", "StagnationInfo"]
