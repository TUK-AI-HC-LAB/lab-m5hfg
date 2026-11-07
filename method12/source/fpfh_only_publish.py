"""Retry atomic publication when Windows temporarily denies a rename."""
import time


def replace_with_retry(temp, target):
    for attempt in range(21):
        try:
            temp.replace(target)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(0.2)
