import subprocess
import os
import time
import sys

# Mocking the run_optimizer_task logic from api.py
def run_optimizer_task_mock(optimizer, source_path):
    print(f"Starting {optimizer} on {source_path}")
    
    cmd = ["python3", "-u", f"{optimizer}.py", source_path]
    
    output_file = "reproduce_output.txt"
    
    print(f"Command: {cmd}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output_lines = []
    
    with open(output_file, 'w') as f:
        for line in process.stdout:
            line_content = line.rstrip()
            output_lines.append(line_content)
            f.write(line)
            f.flush()
            print(f"Captured: {line_content}")
    
    print("Waiting for process to finish...")
    process.wait(timeout=7200)
    print(f"Process finished with return code {process.returncode}")
    
    output = "\n".join(output_lines)
    print("Output captured.")

if __name__ == "__main__":
    # Find a source file
    source_file = "uploads/aa20d656-d533-44a9-8e0f-31c167a328af_matrix_multiply.c"
    if not os.path.exists(source_file):
        print(f"Source file {source_file} not found")
        sys.exit(1)
        
    run_optimizer_task_mock("foga", source_file)
