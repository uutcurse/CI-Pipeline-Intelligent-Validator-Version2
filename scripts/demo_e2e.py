import httpx
import time
import sys
import yaml

def main():
    print("=== CI Workflow Risk Validator E2E Demo ===")
    
    url = "http://127.0.0.1:8080"
    
    try:
        health = httpx.get(f"{url}/health").json()
        print(f"Backend Status: {health['status']}")
    except Exception as e:
        print(f"Failed to connect to backend: {e}")
        sys.exit(1)
        
    workflow_yaml = '''
name: E2E Demo
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make test
'''
    
    print("\nSubmitting workflow...")
    start_time = time.time()
    resp = httpx.post(f"{url}/predict", json={"workflow_yaml": workflow_yaml})
    latency = time.time() - start_time
    
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        sys.exit(1)
        
    data = resp.json()
    
    print("\n--- Workflow Metadata ---")
    print(f"Name: {data['workflow']['name']}")
    print(f"Jobs: {data['workflow']['job_count']}")
    print(f"Steps: {data['workflow']['step_count']}")
    
    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {data['prediction']['predicted_label']}")
    probs = data['prediction']['probabilities']
    print(f"Probabilities:")
    print(f"  LOW:    {probs['LOW'] * 100:.2f}%")
    print(f"  MEDIUM: {probs['MEDIUM'] * 100:.2f}%")
    print(f"  HIGH:   {probs['HIGH'] * 100:.2f}%")
    
    print(f"\nAPI Latency: {latency * 1000:.2f} ms")
    print(f"Inference Engine Latency: {data['inference_duration_ms']:.2f} ms")

if __name__ == '__main__':
    main()
