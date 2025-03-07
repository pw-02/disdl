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
from logger_config import logger


class PrefetchServiceAsync:
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
        self.simulate_time = simulate_time
        self.lambda_name = lambda_name
        self.cache_address = cache_address
        self.lambda_client = boto3.client("lambda")
        self.num_workers = num_workers
        self.lambda_invocations_count = 0
        self.lock = threading.Lock()
        self.prefetch_stop_event = threading.Event()
        self.prefetch_stop_event.set()
        self.queue = Queue()  # Bounded queue to prevent memory overflow

    def start(self):
        """Starts the prefetching service."""
        if not self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already running.")
            return

        logger.info("Starting PrefetchService...")
        self.prefetch_stop_event.clear()
        # Launch worker thread
        worker_thread = threading.Thread(target=self.worker, daemon=True)
        worker_thread.start()

    def stop(self):
        """Stops the prefetching service."""
        if self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already stopped.")
            return

        logger.info("Stopping PrefetchService...")
        self.prefetch_stop_event.set()
        
        self.executor.shutdown(wait=True, cancel_futures=True)
        logger.info(f"Prefetcher stopped. Total Lambda invocations: {self.lambda_invocations_count}")

    def enqueue_batch(self, batch: Batch, payload: dict):
        """Adds a batch to the prefetch queue."""
        try:
            self.queue.put_nowait((batch, payload))
            # logger.debug(f"Batch {batch.batch_id} added to queue. Queue size: {self.queue.qsize()}")
        except Exception as e:
            logger.warning("Prefetch queue is full. Skipping batch prefetch.")

    def worker(self):
        """Worker thread function that processes batches from the queue."""
        while not self.prefetch_stop_event.is_set():
            try:
                batch, payload = self.queue.get(timeout=0.25)
                if self.simulate_time:
                    time.sleep(0.01)  # Simulate a small delay for async processing
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
                self.queue.task_done()
                logger.debug(f"Batch {batch.batch_id}  prefetch requested. Execution time: {response.get('execution_time', 0):.2f}s. Queue size: {self.queue.qsize()}")

            except Empty:
                continue  # No tasks, continue loop
            except Exception as e:
                logger.error(f"Error prefetching batch: {e}")
                self.queue.task_done()
    

class PrefetchServiceEvent:
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
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="PrefetchWorker")

    def start(self):
        """Starts the prefetching service."""
        if not self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already running.")
            return

        logger.info("Starting PrefetchService...")
        self.prefetch_stop_event.clear()
        # Launch worker threads
        worker_thread = threading.Thread(target=self.worker, daemon=True)
        worker_thread.start()

    def stop(self):
        """Stops the prefetching service."""
        if self.prefetch_stop_event.is_set():
            logger.warning("Prefetcher is already stopped.")
            return

        logger.info("Stopping PrefetchService...")
        self.prefetch_stop_event.set()
        
        self.executor.shutdown(wait=True, cancel_futures=True)
        logger.info(f"Prefetcher stopped. Total Lambda invocations: {self.lambda_invocations_count}")

    def enqueue_batch(self, batch: Batch, payload: dict):
        """Adds a batch to the prefetch queue."""
        try:
            self.queue.put_nowait((batch, payload))
            # logger.debug(f"Batch {batch.batch_id} added to queue. Queue size: {self.queue.qsize()}")
        except Exception as e:
            logger.warning("Prefetch queue is full. Skipping batch prefetch.")

    def worker(self):
        """Worker thread function that processes batches from the queue."""
        while not self.prefetch_stop_event.is_set():
            try:
                batch, payload = self.queue.get(timeout=0.5)  # Avoids infinite blocking
                future = self.executor.submit(self.prefetch_batch, batch, payload)
                future.add_done_callback(self.handle_prefetch_result)

                # Dynamically adjust the number of workers based on queue size
                # self.adjust_worker_count()
            except Empty:
                continue  # No tasks, continue loop
    
    def adjust_worker_count(self):
        """Dynamically adjusts the number of worker threads based on the queue size."""
        # If the queue has a large number of items, increase workers
        if self.queue.qsize() > 50:  # This threshold can be adjusted based on your workload
            new_worker_count = min(self.num_workers * 2, 10)  # Limit the upper bound to avoid over-scaling
            if self.executor._max_workers < new_worker_count:
                logger.debug(f"Increasing workers to {new_worker_count}")
                self.executor._max_workers = new_worker_count

        # If the queue is empty, reduce workers to save resources
        elif self.queue.qsize() < 10:  # Adjust threshold as needed
            new_worker_count = max(self.num_workers // 2, 2)  # Limit lower bound to avoid too few workers
            if self.executor._max_workers > new_worker_count:
                logger.debug(f"Decreasing workers to {new_worker_count}")
                self.executor._max_workers = new_worker_count
    
    
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
                response_data = {"success": True, "message": None, "execution_time": execution_time}
            else:
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
                logger.debug(f"Batch {batch.batch_id} successfully prefetched. Execution time: {response.get('execution_time', 0):.2f}s. Queue size: {self.queue.qsize()}")
            else:
                logger.error(f"Batch {batch.batch_id} prefetch failed: {response.get('message', 'Unknown error')}. Execution time: {response.get('execution_time', 0):.2f}s. Queue size: {self.queue.qsize()}")

            self.queue.task_done()
        except concurrent.futures.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Unexpected error in prefetch result handling: {e}")
