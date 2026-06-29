import logging
import threading
import time

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, store, orchestrator, poll_interval_s: float = 0.5):
        self.store = store
        self.orchestrator = orchestrator
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        while not self._stop.is_set():
            job_id = self.store.next_queued()
            if job_id is None:
                time.sleep(self.poll_interval_s)
                continue
            try:
                self.orchestrator.run(job_id)
            except Exception:  # noqa: BLE001 - never let one job kill the worker
                logger.exception("orchestrator.run failed for job %s", job_id)
                time.sleep(self.poll_interval_s)
