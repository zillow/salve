"""Unit tests to make sure execution is aborted for functions that run for too long (timed-out)."""

import time

import pytest

from salve.utils.function_timeout import timeout


def test_timeout() -> None:
    """Ensure that timeout decorator/scope works properly."""
    # should time out
    with pytest.raises(TimeoutError):
        with timeout(seconds=3):
            time.sleep(4)

    # should not time out
    with timeout(seconds=5):
        time.sleep(4)
