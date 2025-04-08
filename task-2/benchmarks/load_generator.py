import argparse
import time
import json
import uuid
import requests
import numpy as np
from typing import List, Dict, Any
import os
import sys
from datasets import load_dataset

from benchmarks.metrics.collector import MetricsCollector

QUERIES_FILE = "data/short_facts_queries.json"

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
        rate_ms = rps / 1000  # miliseconds
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

def run_load_test(args):
    """Run load test with specified parameters"""
    # Generate trace based on specified pattern
    trace = generate_trace(
        pattern=args.pattern,
        rps=args.rps,
        duration=args.duration,
        seed=args.seed
    )
    
    print(f"Generated trace with {len(trace)} requests using {args.pattern} pattern")
    
    queries = load_test_data()
    
    # Initialise metrics collector
    metrics = MetricsCollector()
    
    # Execute requests according to trace
    start_time = time.time()
    
    for i, request_time in enumerate(trace):
        # Calculate delay until next request
        current_time = time.time() - start_time
        delay = request_time / 1000 - current_time
        
        if delay > 0:
            time.sleep(delay)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Select query
        query = queries[i % len(queries)]
        
        # Record request start
        metrics.record_request_start(request_id)
        
        # Send request
        try:
            response = requests.post(
                f"{args.endpoint}/rag",
                json={"query": query},  
                timeout=args.timeout
            )
            
            success = response.status_code == 200
            metrics.record_request_end(request_id, success)
            print(response)
            if not success:
                print(f"Request failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Request error: {e}")
            metrics.record_request_end(request_id, False)
    
    # Save metrics
    metrics.save_results(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run load test for RAG service")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint")
    parser.add_argument("--pattern", choices=["uniform", "poisson", "random"], default="uniform", 
                        help="Pattern for generating request trace")
    parser.add_argument("--rps", type=int, default=5, help="Requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds (duration * rps = total requests)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", type=str, default="load_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    run_load_test(args)