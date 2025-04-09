import argparse
import time
import json
import uuid
import requests
import numpy as np
from typing import List, Dict, Any
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import datetime

from metrics.collector import MetricsCollector
from rag_service.config import Settings

settings = Settings()
QUERIES_FILE = settings.document_queries_file
POLL_INTERVAL = settings.polling_interval


def generate_trace(pattern: str, rps: int, duration: int, seed: int = None) -> List[int]:
    """
    Generate a trace of request timestamps based on the specified pattern.
    
    Args:
        pattern: Distribution pattern ('uniform', 'poisson', or 'random')
        rps: Requests per second
        duration: Duration in seconds
        seed: Random seed for reproducibility
        
    Returns:
        List of timestamps in milliseconds
    """
    if seed is not None:
        np.random.seed(seed)
        
    total_requests = rps * duration
    duration_ms = duration * 1000
    timestamps = []
    
    if total_requests == 0:
        return timestamps
        
    if pattern == "uniform":
        # Evenly distribute requests
        interval = duration_ms / total_requests
        current_time = 0.0
        for _ in range(total_requests):
            timestamp = int(round(current_time))
            timestamp = min(timestamp, duration_ms - 1) 
            timestamps.append(timestamp)
            current_time += interval
            
    elif pattern == "poisson":
        # poisson distributed requests
        rate_ms = rps / 1000 
        intervals = np.random.exponential(1 / rate_ms, total_requests)
        current_time = 0.0
        for i in range(total_requests):
            timestamp = int(round(current_time))
            if timestamp < duration_ms:
                timestamps.append(timestamp)
            current_time += intervals[i]
            
    elif pattern == "random":
        # Random distribution across duration
        timestamps = np.random.randint(0, duration_ms, size=total_requests).tolist()
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
        
    return sorted(timestamps)

def load_test_data():
    """Load queries from pre-saved JSON file""" 
    print(f"Loading queries from {QUERIES_FILE}...")
    if not os.path.exists(QUERIES_FILE):
        print(f"Error: Queries file not found at {QUERIES_FILE}")
        print("Please run the preprocessing script first to generate it.")
        sys.exit(1) # Exit if file is missing

    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        print(f"Loaded {len(queries)} queries from file.")
        return queries
    except Exception as e:
        print(f"An error occurred loading queries: {e}")
        sys.exit(1)

def send_request(endpoint, query, timeout, metrics, request_id, timestamp_data):
    """Send a request and poll for the result"""
    # Record actual request time
    timestamp_data["actual_sent_time"] = time.time()
    
    metrics.record_request_start(request_id)
    
    try:
        # Step 1: Submit the request
        response = requests.post(
            f"{endpoint}/rag",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"Request failed with status code: {response.status_code}")
            metrics.record_request_end(request_id, False)
            timestamp_data["error"] = f"HTTP {response.status_code}"
            timestamp_data["completion_time"] = time.time()
            return
            
        # Extract the request ID from the response
        try:
            response_data = response.json()
            server_request_id = response_data.get("request_id")
            if not server_request_id:
                raise ValueError("No request_id in response")
        except Exception as e:
            print(f"Failed to parse response: {e}")
            metrics.record_request_end(request_id, False)
            timestamp_data["error"] = f"Parse error: {str(e)}"
            timestamp_data["completion_time"] = time.time()
            return
            
        # Step 2: Poll for the result
        start_poll_time = time.time()
        
        while True:
            #check if exceeded timeout
            if time.time() - start_poll_time > timeout:
                print(f"Polling timed out after {timeout} seconds")
                metrics.record_request_end(request_id, False)
                timestamp_data["error"] = "polling_timeout"
                timestamp_data["completion_time"] = time.time()
                return
                
            # Poll for the result
            poll_response = requests.get(
                f"{endpoint}/rag/result/{server_request_id}",
                timeout=10
            )
            
            if poll_response.status_code != 200:
                print(f"Poll failed with status code: {poll_response.status_code}")
                continue
                
            try:
                poll_data = poll_response.json()
                status = poll_data.get("status")
                
                if status == "complete":
                    timestamp_data["completion_time"] = time.time()
                    timestamp_data["result"] = poll_data.get("result")
                    metrics.record_request_end(request_id, True)
                    return
                    
                # If still processing, wait and try again
                time.sleep(POLL_INTERVAL)
                
            except Exception as e:
                print(f"Failed to parse poll response: {e}")
                continue
                
    except Exception as e:
        print(f"Request error: {e}")
        metrics.record_request_end(request_id, False)
        timestamp_data["error"] = str(e)
        timestamp_data["completion_time"] = time.time()

def run_load_test(args):
    """Run load test with specified parameters"""

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate trace based on specified pattern
    trace = generate_trace(
        pattern=args.pattern,
        rps=args.rps,
        duration=args.duration,
        seed=args.seed
    )
    
    print(f"Generated trace with {len(trace)} requests using {args.pattern} pattern")
    
    queries = load_test_data()

    metrics = MetricsCollector()
    
    # Create thread pool
    max_workers = min(32, len(trace)) 
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    start_time = time.time()
    futures = []
    
    timestamp_tracking = []
    
    for i, request_time in enumerate(trace):
        # Calculate delay until next request
        current_time = time.time() - start_time
        delay = request_time / 1000 - current_time
        
        if delay > 0:
            time.sleep(delay)
        
        request_id = str(uuid.uuid4())
        
        query = queries[i % len(queries)]
    
        timestamp_data = {
            "request_id": request_id,
            "query_index": i,
            "planned_time_ms": request_time,
            "scheduled_time": time.time() - start_time
        }
        timestamp_tracking.append(timestamp_data)
        
        # Submit request to thread pool
        future = executor.submit(
            send_request, 
            args.endpoint, 
            query, 
            args.timeout, 
            metrics, 
            request_id,
            timestamp_data  
        )
        futures.append(future)
    
    # Wait for all requests to complete
    for future in futures:
        future.result()
    
    # Calculate actual test duration
    actual_duration = time.time() - start_time
    print(f"Test completed in {actual_duration:.2f} seconds (planned: {args.duration} seconds)")
    
    metrics.save_results(args.output)
    timestamp_file = os.path.splitext(args.output)[0] + "_timestamps.json"
    
    for data in timestamp_tracking:
        if "actual_sent_time" in data:
            data["actual_sent_time_relative"] = data["actual_sent_time"] - start_time
            data["scheduling_delay_ms"] = (data["scheduled_time"] - data["planned_time_ms"]/1000) * 1000
        
        if "completion_time" in data and "actual_sent_time" in data:
            data["execution_time"] = data["completion_time"] - data["actual_sent_time"]
    
    # Save detailed timestamp data
    with open(timestamp_file, 'w') as f:
        json.dump({
            "test_info": {
                "pattern": args.pattern,
                "rps": args.rps,
                "duration": args.duration,
                "actual_duration": actual_duration,
                "timestamp": datetime.datetime.now().isoformat()
            },
            "requests": timestamp_tracking
        }, f, indent=2)
    
    print(f"Detailed timestamp data saved to {timestamp_file}")
    
    # summary output stats
    results = metrics.calculate_metrics()
    print("\nTest Results Summary:")
    print(f"Total Requests:      {results['total_requests']}")
    print(f"Successful Requests: {results['successful_requests']}")
    print(f"Failed Requests:     {results['failed_requests']}")
    print(f"Actual Throughput:   {results['throughput']:.2f} req/sec")
    print(f"Mean Latency:        {results['latency']['mean']:.2f} sec")
    print(f"P95 Latency:         {results['latency']['p95']:.2f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run load test for RAG service")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint")
    parser.add_argument("--pattern", choices=["uniform", "poisson", "random"], default="uniform", 
                        help="Pattern for generating request trace")
    parser.add_argument("--rps", type=int, default=5, help="Requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds (duration * rps = total requests)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", type=str, default="benchmarks/results/load_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    run_load_test(args)