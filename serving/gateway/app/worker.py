import threading
import time


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

    def _loop(self):
        while not self._stop.is_set():
            job_id = self.store.next_queued()
            if job_id is None:
                time.sleep(self.poll_interval_s)
                continue
            self.orchestrator.run(job_id)
