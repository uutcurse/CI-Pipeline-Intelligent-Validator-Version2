import httpx
import time
import sys

def main():
    print("Running Smoke Test...")
    yaml_in = '''
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo 'Smoke Test'
'''
    
    url = "http://127.0.0.1:8080"
    
    # Wait for API to come up
    for _ in range(10):
        try:
            resp = httpx.get(f"{url}/health")
            if resp.status_code == 200:
                print("API is UP.")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("API failed to start.")
        sys.exit(1)
        
    print("Checking model info...")
    resp = httpx.get(f"{url}/model-info")
    assert resp.status_code == 200
    print("Model info OK:", resp.json()["model_id"])

    print("Sending predict request...")
    resp = httpx.post(f"{url}/predict", json={"workflow_yaml": yaml_in})
    assert resp.status_code == 200, f"Failed: {resp.text}"
    
    data = resp.json()
    assert "prediction" in data
    assert "probabilities" in data["prediction"]
    
    probs = data["prediction"]["probabilities"]
    print("Probabilities:", probs)
    
    assert sum(probs.values()) > 0.99
    print("Smoke test SUCCESS.")

if __name__ == '__main__':
    main()
