import threading
import time
import json
import boto3
import concurrent.futures
import logging
from queue import Queue, Empty
from datetime import datetime, timezone
from batch import Batch  # Ensure these are properly defined

from cache_status import CacheStatus
# from logger_config import configure_logger
import logging
# logger = configure_logger()

logger = logging.getLogger()

import redis

class PrefetchServiceAsync:
    def __init__(self, lambda_name: str, cache_address: str, simulate_time: float = None):

        self.simulate_time = simulate_time
        self.lambda_name = lambda_name
        self.cache_address = cache_address
        self.lambda_client = boto3.client("lambda")
        self.lambda_invocations_count = 0
        self.lock = threading.Lock()
        self.prefetch_stop_event = threading.Event()
        self.prefetch_stop_event.set()
        self.prefetch_queue = Queue()  # Bounded queue to prevent memory overflow
        self.prefetch_thread = None
        self.status_check_queue = Queue()
        self.status_checker_thread = None

    
    def start(self):
        """Starts the prefetching service."""
        if not self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already running.")
            return

        logger.info("Starting PrefetchService...")
        self.prefetch_stop_event.clear()
        # Launch worker thread
        self.prefetch_thread = threading.Thread(target=self.prefetch_worker, daemon=True)
        self.prefetch_thread .start()

        # Launch status checker thread
        self.status_checker_thread = threading.Thread(target=self.status_checker_worker, daemon=True)
        self.status_checker_thread.start()

    def stop(self):
        """Stops the prefetching service."""
        if self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already stopped.")
            return
        
        logger.info("Stopping PrefetchService...")
        self.prefetch_stop_event.set()
        logger.info(f"Prefetcher stopped")
    
    def enqueue_batch(self, batch: Batch, payload: dict):
        """Adds a batch to the prefetch queue."""
        try:
            self.prefetch_queue.put_nowait((batch, payload))
            logger.debug(f"Batch {batch.batch_id} added to queue. Queue size: {self.prefetch_queue.qsize()}")
        except Exception as e:
            logger.warning("Prefetch queue is full. Skipping batch prefetch.")

    def prefetch_worker(self):
        """Worker thread function that processes batches from the queue."""
        while not self.prefetch_stop_event.is_set():
            try:
                batch, payload = self.prefetch_queue.get(timeout=0.25)
                batch:Batch = batch
                if self.simulate_time:
                    time.sleep(self.simulate_time)  # Simulate a small delay for async processing
                    self.lambda_invocations_count += 1
                else:
                    response = self.lambda_client.invoke(
                        FunctionName=self.lambda_name,
                        InvocationType="Event",  # Asynchronous invocation
                        Payload=payload
                        )
                
                batch.set_cache_status(CacheStatus.CACHING_IN_PROGRESS)

                with self.lock:
                    self.lambda_invocations_count += 1
                    batch.set_last_accessed_time()
                    batch.prefetched_time_utc = datetime.now(timezone.utc)
                
                self.status_check_queue.put(batch)  # Add to polling queue
                self.prefetch_queue.task_done()
                logger.debug(f"Batch {batch.batch_id} prefetch requested. Queue size: {self.prefetch_queue.qsize()}")
            except Empty:
                continue  # No tasks, continue loop
            except Exception as e:
                logger.error(f"Error prefetching batch: {e}")
                self.prefetch_queue.task_done()

    
    def status_checker_worker(self):
        redis_client = redis.StrictRedis.from_url(self.cache_address)
        while not self.prefetch_stop_event.is_set():
            try:
                batch:Batch = self.status_check_queue.get(timeout=0.25)
                batch_id = batch.batch_id
                if redis_client.exists(f"{batch_id}"):
                    batch.set_cache_status(CacheStatus.CACHED)
                    logger.debug(f"Batch {batch_id} is now ready in Redis.")
                else:
                    # Not ready yet, re-enqueue for later checking
                    self.status_check_queue.put(batch)
                    time.sleep(0.25)  # backoff
                self.status_check_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error checking batch status: {e}")
