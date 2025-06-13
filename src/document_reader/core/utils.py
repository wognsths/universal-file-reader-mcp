"""Utility helpers for processors."""

from __future__ import annotations

import asyncio
import concurrent.futures
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
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=seconds)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(
                            f"Processing timeout after {seconds} seconds"
                        )

            return wrapper

    return decorator
