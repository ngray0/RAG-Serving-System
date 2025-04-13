import argparse
import time
import json
import uuid
import requests
import numpy as np
import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

from metrics.collector import MetricsCollector
from rag_service.config import Settings

settings = Settings()
QUERIES_FILE = settings.document_queries_file
POLL_INTERVAL = settings.polling_interval

def generate_trace(pattern: str, rps: int, duration: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        
    total_requests = rps * duration
    duration_ms = duration * 1000
    timestamps = []
    
    if total_requests == 0:
        return timestamps
        
    if pattern == "uniform":
        interval = duration_ms / total_requests
        current_time = 0.0
        for _ in range(total_requests):
            timestamp = int(round(current_time))
            timestamp = min(timestamp, duration_ms - 1) 
            timestamps.append(timestamp)
            current_time += interval
            
    elif pattern == "poisson":
        rate_ms = rps / 1000 
        intervals = np.random.exponential(1 / rate_ms, total_requests)
        current_time = 0.0
        for i in range(total_requests):
            timestamp = int(round(current_time))
            if timestamp < duration_ms:
                timestamps.append(timestamp)
            current_time += intervals[i]
            
    elif pattern == "random":
        timestamps = np.random.randint(0, duration_ms, size=total_requests).tolist()
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
        
    return sorted(timestamps)

def load_test_data():
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries

async def send_request_async(endpoint, query, timeout, metrics, request_id, timestamp_data):
    metrics.record_request_start(request_id)
    
    loop = asyncio.get_event_loop()
    
    try:
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{endpoint}/rag",
                json={"query": query},
                timeout=30
            )
        )
        
        if response.status_code != 200:
            metrics.record_request_end(request_id, False)
            return
            
        try:
            response_data = response.json()
            server_request_id = response_data.get("request_id")
            if not server_request_id:
                raise ValueError("No request_id in response")
        except Exception:
            metrics.record_request_end(request_id, False)
            return
            
        start_poll_time = time.time()
        
        while True:
            if time.time() - start_poll_time > timeout:
                metrics.record_request_end(request_id, False)
                return
                
            # Poll for the result
            poll_response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    f"{endpoint}/rag/result/{server_request_id}",
                    timeout=10
                )
            )
            
            if poll_response.status_code != 200:
                continue
                
            try:
                poll_data = poll_response.json()
                status = poll_data.get("status")
                
                if status == "complete":
                    metrics.record_request_end(request_id, True)
                    return
                    
                await asyncio.sleep(POLL_INTERVAL)
                
            except Exception:
                continue
                
    except Exception:
        metrics.record_request_end(request_id, False)

class AsyncRequestDispatcher:
    def __init__(self, endpoint, timeout, metrics, max_concurrency=200):
        self.endpoint = endpoint
        self.timeout = timeout
        self.metrics = metrics
        self.dispatch_queue = asyncio.Queue()
        self.max_concurrency = max_concurrency
        self.workers = []
        
    async def worker(self):
        while True:
            try:
                req = await self.dispatch_queue.get()
                await send_request_async(
                    self.endpoint,
                    req["query"],
                    self.timeout,
                    self.metrics,
                    req["request_id"],
                    req["timestamp_data"]
                )
                self.dispatch_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.dispatch_queue.task_done()
                print(f"Worker error: {e}")
    
    async def start_workers(self):
        self.workers = [asyncio.create_task(self.worker()) 
                        for _ in range(self.max_concurrency)]
                
    async def stop_workers(self):
        for worker in self.workers:
            worker.cancel()
        
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
    
    async def schedule_request(self, req):
        await self.dispatch_queue.put(req)
    
    async def wait_completion(self):
        await self.dispatch_queue.join()

async def run_async_load_test(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    trace = generate_trace(
        pattern=args.pattern,
        rps=args.rps,
        duration=args.duration,
        seed=args.seed
    )
    
    queries = load_test_data()
    metrics = MetricsCollector()
    
    # Pre-generate all request data before timing starts
    all_requests = []
    
    for i, request_time in enumerate(trace):
        request_id = str(uuid.uuid4())
        query = queries[i % len(queries)]
        
        timestamp_data = {"request_id": request_id}
        
        all_requests.append({
            "time_ms": request_time,
            "request_id": request_id,
            "query": query,
            "timestamp_data": timestamp_data
        })
    
    # Create and start the request dispatcher with worker pool
    dispatcher = AsyncRequestDispatcher(
        endpoint=args.endpoint,
        timeout=args.timeout,
        metrics=metrics,
        max_concurrency=min(200, len(trace))
    )
    
    await dispatcher.start_workers()
    
    # Start the actual test
    start_time = time.time()
    
    # Schedule the requests at their precise times
    for req in all_requests:
        # Calculate when this request should be sent (absolute time)
        target_abs_time = start_time + (req["time_ms"] / 1000)
        now = time.time()
        
        # Wait until the scheduled time
        if now < target_abs_time:
            await asyncio.sleep(target_abs_time - now)
        
        # Queue the request for execution by a worker
        await dispatcher.schedule_request(req)
    
    # Wait for all requests to complete
    await dispatcher.wait_completion()
    
    # Stop all workers
    await dispatcher.stop_workers()
    
    # Save results
    metrics.save_results(args.output)

def run_load_test(args):
    asyncio.run(run_async_load_test(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run load test for RAG service")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint")
    parser.add_argument("--pattern", choices=["uniform", "poisson", "random"], default="uniform", 
                        help="Pattern for generating request trace")
    parser.add_argument("--rps", type=int, default=50, help="Requests per second")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds (duration * rps = total requests)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", type=str, default="benchmarks/results/load_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    run_load_test(args)