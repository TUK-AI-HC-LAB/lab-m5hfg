"""Bounded background H2D/concat on a dedicated CUDA stream."""
import queue,threading
import torch

class CUDAPrefetch:
    def __init__(self,loader):
        self.loader=iter(loader);self.queue=queue.Queue(2);self.stop=threading.Event()
        self.stream=torch.cuda.Stream();self.thread=threading.Thread(target=self._produce,daemon=True);self.thread.start()
    def _put(self,item):
        while not self.stop.is_set():
            try:self.queue.put(item,timeout=.1);return True
            except queue.Full:pass
        return False
    def _produce(self):
        try:
            for data in self.loader:
                if self.stop.is_set():break
                with torch.cuda.stream(self.stream):
                    x=torch.cat([p.cuda(non_blocking=True) for p in data]).to(memory_format=torch.channels_last)
                    event=torch.cuda.Event();event.record(self.stream)
                if not self._put((x,event)):break
            self._put(None)
        except BaseException as error:self._put(error)
    def __iter__(self):return self
    def __next__(self):
        item=self.queue.get()
        if item is None:raise StopIteration
        if isinstance(item,BaseException):raise item
        x,event=item;stream=torch.cuda.current_stream();stream.wait_event(event);x.record_stream(stream)
        return x
    def close(self):
        self.stop.set();self.thread.join(timeout=5)
