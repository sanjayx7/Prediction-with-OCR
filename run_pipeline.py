import os
import sys
import subprocess

def run_script(script_path):
    print(f"\n==========================================")
    print(f"Running script: {script_path}")
    print(f"==========================================")
    
    python_exe = sys.executable
    result = subprocess.run([python_exe, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Script {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"SUCCESS: Script {script_path} completed successfully.")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Run Section A: Premium Prediction Pipeline
    section_a_script = os.path.join(root_dir, 'section_a', 'train_predict.py')
    run_script(section_a_script)
    
    # 2. Run Section B: OCR Text Extraction Pipeline
    section_b_script = os.path.join(root_dir, 'section_b', 'extract_text.py')
    run_script(section_b_script)
    
    # 3. Launch FastAPI App (Section C)
    print(f"\n==========================================")
    print(f"Starting FastAPI Web Server (Section C)...")
    print(f"Access the Dashboard at: http://127.0.0.1:8000")
    print(f"==========================================")
    
    # Add section_c to python path so uvicorn can find the app module
    sys.path.append(os.path.join(root_dir, 'section_c'))
    
    try:
        import uvicorn
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
    except ImportError:
        print("ERROR: uvicorn is not installed in the current environment.")
        print("Please install requirements: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\nWeb server stopped by user.")

if __name__ == '__main__':
    main()
