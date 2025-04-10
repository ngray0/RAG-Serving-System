import time
import os
import redis
import json
from kubernetes import client, config

# Initialize Kubernetes client
config.load_incluster_config()
k8s_apps_v1 = client.AppsV1Api()

# Redis connection
redis_client = redis.from_url(os.environ.get("REDIS_URL", "redis://redis-service:6379/0"))

# Config
NAMESPACE = os.environ.get("NAMESPACE", "default")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME", "rag-service")
MIN_REPLICAS = int(os.environ.get("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.environ.get("MAX_REPLICAS", "4"))
QUEUE_KEY = os.environ.get("QUEUE_KEY", "rag_service:requests")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "15"))  # seconds
WAIT_THRESHOLD = float(os.environ.get("WAIT_THRESHOLD", "5.0"))  # seconds
QUEUE_SIZE_PER_REPLICA = int(os.environ.get("QUEUE_SIZE_PER_REPLICA", "30"))

def get_queue_metrics():
    # Get queue size
    queue_size = redis_client.llen(QUEUE_KEY)
    
    # Get wait time of oldest item
    wait_time = 0
    if queue_size > 0:
        try:
            oldest_item = redis_client.lindex(QUEUE_KEY, 0)
            if oldest_item:
                item_data = json.loads(oldest_item)
                if "timestamp" in item_data:
                    wait_time = time.time() - item_data["timestamp"]
        except Exception as e:
            print(f"Error calculating wait time: {e}")
    
    return queue_size, wait_time

def scale_deployment(desired_replicas):
    current_replicas = k8s_apps_v1.read_namespaced_deployment(
        name=DEPLOYMENT_NAME, namespace=NAMESPACE
    ).spec.replicas
    
    if current_replicas == desired_replicas:
        return False
    
    print(f"Scaling {DEPLOYMENT_NAME} from {current_replicas} to {desired_replicas} replicas")
    
    try:
        k8s_apps_v1.patch_namespaced_deployment_scale(
            name=DEPLOYMENT_NAME,
            namespace=NAMESPACE,
            body={"spec": {"replicas": desired_replicas}}
        )
        return True
    except Exception as e:
        print(f"Error scaling deployment: {e}")
        return False

def autoscale_loop():
    while True:
        try:
            queue_size, wait_time = get_queue_metrics()
            print(f"Queue metrics - Size: {queue_size}, Wait time: {wait_time:.2f}s")
            
            # Calculate desired replicas based on queue size
            size_based_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, 
                                     (queue_size // QUEUE_SIZE_PER_REPLICA) + 1))
            
            # If wait time is too high, add more replicas
            if wait_time > WAIT_THRESHOLD:
                # Scale up by one more replica if wait time is high
                desired_replicas = min(MAX_REPLICAS, size_based_replicas + 1)
            else:
                desired_replicas = size_based_replicas
            
            scale_deployment(desired_replicas)
            
        except Exception as e:
            print(f"Error in autoscaler loop: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    print("Starting RAG Service Autoscaler")
    autoscale_loop()