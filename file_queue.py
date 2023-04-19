import time, random, marshal, os

MAX_AGE = 5

class FileQueue:
    def __init__(self, est_time=60, id=None):
        queue:list = self.load()
        self.id = random.randint(0, 2**16) if id is None else id
        self.est_time = est_time
        self.start = time.time()
        queue.append((self.id, self.est_time, self.start, self.start))
        self.save(queue)
    
    def load(self) -> list:
        if not os.path.exists("queue"):
            self.save([])
        
        try:
            with open("queue", "rb") as f:
                return marshal.load(f)
        except EOFError:
            time.sleep(random.random())
            return self.load()

    def save(self, queue:list):
        try:
            with open("queue", "wb") as f:
                marshal.dump(queue, f)
        except OSError:
            time.sleep(random.random())
            self.save(queue)
    
    def heartbeat(self):
        queue = self.load()
        for i, q in enumerate(queue):
            if q[0] == self.id:
                queue[i] = (self.id, self.est_time, self.start, time.time())
                break
        self.save(queue)

    def should_run(self) -> bool:
        queue = self.load()
        queue = [q for q in queue if q[3] > time.time() - MAX_AGE and q[2] < self.start]
        queue.sort(key=lambda x: x[2])
        if len(queue) == 0:
            return True
        return queue[0][0] == self.id # First in queue
    
    def update_est_time(self, est_time:float):
        queue = self.load()
        for i, q in enumerate(queue):
            if q[0] == self.id:
                queue[i] = (self.id, est_time, self.start, time.time())
                break
        self.save(queue)
    
    def get_queue_len(self) -> int:
        queue = self.load()
        count = 0
        for q in queue:
            if q[3] > time.time() - MAX_AGE and q[2] < self.start:
                count += 1
        return count
    
    def get_queue_est_time(self) -> float:
        queue = self.load()
        count = 0
        for q in queue:
            if q[3] > time.time() - MAX_AGE and q[2] < self.start:
                count += q[1]
        return count
    
    def quit(self):
        queue = self.load()
        for i, q in enumerate(queue):
            if q[0] == self.id:
                del queue[i]
                break
        self.save(queue)

    def __del__(self):
        self.quit()

if __name__ == '__main__':
    import threading

    def test(worker_id):
        q = FileQueue()
        
        # Wait to be first in queue
        while not q.should_run():
            time.sleep(1)
            q.heartbeat()
        
        # Do stuff
        print(f"Worker {worker_id} started")
        for i in range(10):
            time.sleep(1)
            q.heartbeat()
            print(f"Worker {worker_id} progress: {i + 1}/10")
        
        # Leave queue
        print(f"Worker {worker_id} finished")
        q.quit()

    for i in range(5):
        threading.Thread(target=test, args=(i,)).start()
        time.sleep(0.123)

