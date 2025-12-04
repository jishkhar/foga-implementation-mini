import requests
import time
import threading
import json
import sys
import os

BASE_URL = "http://localhost:8000"

def stream_output(job_id):
    print(f"Connecting to stream for {job_id}...")
    try:
        with requests.get(f"{BASE_URL}/jobs/{job_id}/stream", stream=True) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(f"STREAM: {decoded_line}")
                    if "event: close" in decoded_line:
                        break
    except Exception as e:
        print(f"Stream error: {e}")

def main():
    # 1. Submit Job
    print("Submitting job...")
    source_file = "uploads/aa20d656-d533-44a9-8e0f-31c167a328af_matrix_multiply.c"
    if not os.path.exists(source_file):
        # Try to find any .c file in uploads
        files = [f for f in os.listdir("uploads") if f.endswith(".c")]
        if files:
            source_file = os.path.join("uploads", files[0])
        else:
            print("No source file found")
            return

    with open(source_file, 'rb') as f:
        files = {'source_file': f}
        response = requests.post(f"{BASE_URL}/optimize/foga", files=files)
    
    if response.status_code != 200:
        print(f"Failed to submit job: {response.text}")
        return
        
    job = response.json()
    job_id = job['job_id']
    print(f"Job submitted: {job_id}")
    
    # 2. Start Streaming in background
    stream_thread = threading.Thread(target=stream_output, args=(job_id,))
    stream_thread.start()
    
    # 3. Poll Status
    print("Polling status...")
    while True:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        if response.status_code != 200:
            print(f"Poll failed: {response.status_code}")
            break
            
        status_data = response.json()
        status = status_data['status']
        print(f"Status: {status}")
        
        if status in ['completed', 'failed']:
            print("Job finished!")
            print(json.dumps(status_data, indent=2))
            break
            
        time.sleep(2)
        
    stream_thread.join(timeout=5)

if __name__ == "__main__":
    main()
