from __future__ import print_function
from threading import Thread, Event
from Queue import Queue, Empty


class DataThread(Thread):
    def __init__(self, data_pipeline, max_queue_size=8, **get_batch_kwargs):
        super(DataThread, self).__init__(name='neuralnilm-data-process')
        self._stop = Event()
        self._queue = Queue(maxsize=max_queue_size)
        self.data_pipeline = data_pipeline
        self._get_batch_kwargs = get_batch_kwargs

    def run(self):
        while not self._stop.is_set():
            batch = self.data_pipeline.get_batch(**self._get_batch_kwargs)
            self._queue.put(batch)

    def get_batch(self, timeout=30):
        if self.is_alive():
            return self._queue.get(timeout=timeout)
        else:
            raise RuntimeError("Process is not running!")

    def stop(self):
        self._stop.set()
        try:
            self._queue.get(block=False)
        except Empty:
            pass
        self.join()
