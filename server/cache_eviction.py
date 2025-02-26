import threading
from typing import  Dict, Tuple
import time
from logger_config import logger
from job import DLTJob
import redis

class CacheEvictionService:
    def __init__(self, cache_address: str, jobs:Dict[str, DLTJob], keep_alive_time_threshold:int = 1000, simulate_keep_alvive: bool = False):
        self.cache_address = cache_address
        self.cache_eviction_stop_event = threading.Event()  # Event to signal stopping
        self.keep_alive_time_threshold = keep_alive_time_threshold
        self.jobs:Dict[str, DLTJob] = jobs
        self.cache_eviction_stop_event.set()
        self.simulate_keep_alvive = simulate_keep_alvive
        self.redis_client = None
        self.lock = threading.Lock()

    def start_cache_evictor(self):
        self.cache_eviction_stop_event.clear()
        keep_alive_thread = threading.Thread(target=self._keep_alive_process)
        keep_alive_thread.daemon = True
        keep_alive_thread.start()
     
    def stop_cache_evictor(self):
        if not self.cache_eviction_stop_event.is_set():
            self.cache_eviction_stop_event.set()
            logger.info(f"Cache eviction service stopped")

        

    def _keep_alive_process(self):
        while not self.cache_eviction_stop_event.is_set():  # Check for stop signal
            try:
                cache_host, cache_port = self.cache_address.split(":")

                if self.redis_client is None and not self.simulate_keep_alvive:
                    self.redis_client = redis.StrictRedis(host=cache_host, port=cache_port)

                #Take a snapshot
                with self.lock:
                    jobs_snapshot = list(self.jobs.values())

                for job in jobs_snapshot:
                    if job.total_steps <= 1:
                        continue
                    # job_batches_snapshot = list(job.future_batches.values())
                    for batch in job.future_batches.values():
                        
                        if batch.time_since_last_access() > self.keep_alive_time_threshold:
                            try:
                                if self.simulate_keep_alvive:
                                        batch.set_cache_status(is_cached=True)
                                        batch.set_last_accessed_time()
                                else:
                                    if self.redis_client.get(batch.batch_id):  # Check if the batch is still cached
                                            batch.set_last_accessed_time()  # Update the last accessed time
                                            batch.set_cache_status(is_cached=True)
                                    else:
                                            # If the batch is not in Redis, mark it as not cached
                                            batch.set_cache_status(is_cached=False)
                                            logger.warning(f"Batch '{batch.batch_id}' is not in cache, setting is_cached to False.")
                            except Exception as e:
                                    logger.error(f"Error keeping batch '{batch.batch_id}' alive: {e}")
                                    batch.set_cache_status(is_cached=False)
                time.sleep(5)  # Sleep for a short while before checking the queue again
            
            except Exception as e:
                logger.error(f"Unexpected error in cache eviction process: {e}", exc_info=True)