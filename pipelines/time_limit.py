"""
Timeout-hulp voor pipelineverwerking.

De pipelines draaien op macOS/Linux; daarom gebruiken we SIGALRM om een
vastgelopen factuurverwerking hard af te breken.
"""

from __future__ import annotations

from contextlib import contextmanager
import signal


FACTUUR_TIMEOUT_SECONDEN = 15 * 60


class FactuurTimeout(TimeoutError):
    pass


@contextmanager
def factuur_timeout(seconden: int = FACTUUR_TIMEOUT_SECONDEN):
    def _handler(signum, frame):
        raise FactuurTimeout(f"Factuurverwerking duurde langer dan {seconden} seconden")

    vorige_handler = signal.getsignal(signal.SIGALRM)
    vorige_timer = signal.setitimer(signal.ITIMER_REAL, seconden)
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, vorige_handler)
        if vorige_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, vorige_timer[0], vorige_timer[1])
