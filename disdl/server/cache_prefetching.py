import threading
import time
import json
import boto3
import concurrent.futures
import logging
from queue import Queue, Empty
from datetime import datetime, timezone
from batch import Batch, CacheStatus  # Ensure these are properly defined
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class PrefetchService:
    def __init__(self, lambda_name: str, cache_address: str, simulate_time: float = None, num_workers: int = 4):
        """
        Initializes the PrefetchService.

        Args:
            lambda_name (str): Name of the AWS Lambda function.
            cache_address (str): Cache storage location (e.g., S3, Redis).
            simulate_time (float, optional): Simulated prefetch time for testing.
            num_workers (int, optional): Number of parallel workers for prefetching. Default is 4.
            queue_size (int, optional): Maximum size of the prefetch queue.
        """
        self.lambda_name = lambda_name
        self.cache_address = cache_address
        self.lambda_client = boto3.client("lambda")
        self.num_workers = num_workers
        self.simulate_time = simulate_time
        self.lambda_invocations_count = 0
        self.lock = threading.Lock()
        self.prefetch_stop_event = threading.Event()
        self.prefetch_stop_event.set()
        self.queue = Queue()  # Bounded queue to prevent memory overflow
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

    def start(self):
        """Starts the prefetching service."""
        if not self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already running.")
            return

        logger.info("Starting PrefetchService...")
        self.prefetch_stop_event.clear()

        # Launch worker threads
        for _ in range(self.num_workers):
            worker_thread = threading.Thread(target=self.worker, daemon=True)
            worker_thread.start()

    def stop(self):
        """Stops the prefetching service."""
        if self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already stopped.")
            return

        logger.info("Stopping PrefetchService...")
        self.prefetch_stop_event.set()
        
        self.executor.shutdown(wait=False, cancel_futures=True)
        logger.info(f"Prefetcher stopped. Total Lambda invocations: {self.lambda_invocations_count}")

    def enqueue_batch(self, batch: Batch, payload: dict):
        """Adds a batch to the prefetch queue."""
        try:
            self.queue.put_nowait((batch, payload))
            logger.debug(f"Batch {batch.batch_id} added to queue. Queue size: {self.queue.qsize()}")
        except Exception as e:
            logger.warning("Prefetch queue is full. Skipping batch prefetch.")

    def worker(self):
        """Worker thread function that processes batches from the queue."""
        while not self.prefetch_stop_event.is_set():
            try:
                batch, payload = self.queue.get(timeout=0.5)  # Avoids infinite blocking
                future = self.executor.submit(self.prefetch_batch, batch, payload)
                future.add_done_callback(self.handle_prefetch_result)
            except Empty:
                continue  # No tasks, continue loop

    def prefetch_batch(self, batch: Batch, payload: dict):
        """
        Prefetches a batch using AWS Lambda or simulated time.

        Args:
            batch (Batch): The batch to prefetch.
            payload (dict): Payload for the Lambda invocation.

        Returns:
            tuple: (batch, response_dict)
        """
        start_time = time.perf_counter()

        try:
            if self.simulate_time:
                time.sleep(self.simulate_time)
                execution_time = time.perf_counter() - start_time
                # logger.info(f"Simulated prefetch of batch {batch.batch_id} in {execution_time:.2f}s")
                return batch, {"success": True, "message": None, "execution_time": execution_time}

            response = self.lambda_client.invoke(
                FunctionName=self.lambda_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload)
            )
            response_data = json.loads(response["Payload"].read().decode("utf-8"))

            with self.lock:
                self.lambda_invocations_count += 1

            return batch, response_data

        except Exception as e:
            logger.error(f"Error invoking Lambda for batch {batch.batch_id}: {e}")
            return batch, {"success": False, "message": str(e)}

    def handle_prefetch_result(self, future):
        """
        Handles the result of a completed prefetch operation.

        Args:
            future (concurrent.futures.Future): The future representing the completed task.
        """
        try:
            batch, response = future.result()  # Unpack result
            if response.get("success", False):
                batch.set_cache_status(CacheStatus.CACHED)
                batch.set_last_accessed_time()
                batch.prefetched_time_utc = datetime.now(timezone.utc)
                logger.info(f"Batch {batch.batch_id} successfully prefetched. Execution time: {response.get('execution_time', 0):.2f}s")
            else:
                logger.error(f"Batch {batch.batch_id} prefetch failed: {response.get('message', 'Unknown error')}. Execution time: {response.get('execution_time', 0):.2f}s")

            self.queue.task_done()


        except Exception as e:
            logger.error(f"Unexpected error in prefetch result handling: {e}")
