"""Shared retry helper for LLM calls across all pipelines."""

import time
import logging

logger = logging.getLogger("control")

_RETRY_WAIT  = 30
_MAX_RETRIES = 5


def call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on capacity/rate-limit/timeout errors.

    Timeouts are included because the lab server has shown transient slow
    windows (a call can time out at the full request timeout, then succeed
    in seconds moments later) — safe to retry now that the client itself has
    max_retries=0 and a bounded per-call timeout (config.get_llm), so each
    retry attempt here is still capped, not multiplied.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_retryable = any(
                k in msg for k in
                ("capacity", "rate limit", "429", "503", "overloaded", "timeout", "timed out")
            )
            if is_retryable and attempt < _MAX_RETRIES:
                logger.warning(
                    f"Provider error (attempt {attempt}/{_MAX_RETRIES}): {exc} "
                    f"— retrying in {_RETRY_WAIT}s"
                )
                time.sleep(_RETRY_WAIT)
            else:
                raise
    return None
