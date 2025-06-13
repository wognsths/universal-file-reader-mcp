"""Utility helpers for processors."""

from __future__ import annotations

import signal
import asyncio
from functools import wraps


def with_timeout(seconds: int):
    """Decorator to abort a function if it exceeds the given timeout."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.wait_for(func(*args, **kwargs), seconds)

            return async_wrapper
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                def handler(signum, frame):
                    raise TimeoutError(f"Processing timeout after {seconds} seconds")

                signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)

            return wrapper

    return decorator
