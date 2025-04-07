import time
import json
import numpy as np
from typing import Dict, List, Any

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.request_times = {}  # Map of request_id to (start_time, end_time)
        self.latencies = []
        self.success_count = 0
        self.error_count = 0
        
    def record_request_start(self, request_id: str):
        """Record when a request starts"""
        self.request_times[request_id] = (time.time(), None)
        
    def record_request_end(self, request_id: str, success: bool):
        """Record when a request completes"""
        if request_id in self.request_times:
            start_time, _ = self.request_times[request_id]
            end_time = time.time()
            self.request_times[request_id] = (start_time, end_time)
            
            # Calculate latency
            latency = end_time - start_time
            self.latencies.append(latency)
            
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate throughput and latency metrics"""
        total_time = time.time() - self.start_time
        total_requests = self.success_count + self.error_count
        
        metrics = {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "total_time": total_time
        }
        
        # Throughput = Number of requests processed / Total processing time
        if total_time > 0:
            metrics["throughput"] = total_requests / total_time
            metrics["throughput_successful"] = self.success_count / total_time
        else:
            metrics["throughput"] = 0
            metrics["throughput_successful"] = 0
        
        # Latency statistics
        if self.latencies:
            metrics["latency"] = {
                "min": min(self.latencies),
                "max": max(self.latencies),
                "mean": sum(self.latencies) / len(self.latencies),
                "p50": np.percentile(self.latencies, 50),
                "p95": np.percentile(self.latencies, 95),
                "p99": np.percentile(self.latencies, 99)
            }
        else:
            metrics["latency"] = {
                "min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0
            }
            
        return metrics
    
    def save_results(self, filename: str):
        """Save metrics to a JSON file"""
        metrics = self.calculate_metrics()
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\nPerformance Results:")
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Throughput: {metrics['throughput']:.2f} requests/second")
        
        if self.latencies:
            print(f"Average Latency: {metrics['latency']['mean']:.4f} seconds")
            print(f"95th Percentile Latency: {metrics['latency']['p95']:.4f} seconds")
            
        print(f"Results saved to {filename}")
        return metrics
