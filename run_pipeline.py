import os
import sys
import subprocess

def get_python_exe(root_dir):
    local_venv = os.path.join(root_dir, "venv", "Scripts", "python.exe")
    if os.path.exists(local_venv):
        return local_venv
    ext_venv = r"C:\Users\ACER\ML\venv\Scripts\python.exe"
    if os.path.exists(ext_venv):
        return ext_venv
    return sys.executable

def run_script(script_path, root_dir):
    print(f"\n==========================================")
    print(f"Running script: {script_path}")
    print(f"==========================================")

    python_exe = get_python_exe(root_dir)
    print(f"Using Python executable: {python_exe}")
    
    result = subprocess.run([python_exe, script_path], capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: Script {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"SUCCESS: Script {script_path} completed successfully.")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Run Section A: Premium Prediction Pipeline (Model Training & Metric Outputs)
    section_a_script = os.path.join(root_dir, 'section_a', 'train_predict.py')
    run_script(section_a_script, root_dir)

    # 2. Launch FastAPI App (Section C)
    print(f"\n==========================================")
    print(f"Starting FastAPI Web Server (Section C)...")
    print(f"Access the Dashboard at: http://127.0.0.1:8000")
    print(f"==========================================")

    python_exe = get_python_exe(root_dir)
    print(f"Using Python executable: {python_exe}")
    section_c_dir = os.path.join(root_dir, 'section_c')

    try:
        # --reload ensures uvicorn always re-reads the latest code on each request
        subprocess.run(
            [python_exe, "-m", "uvicorn", "app:app",
             "--host", "127.0.0.1", "--port", "8000", "--reload"],
            cwd=section_c_dir,
            capture_output=False,
        )
    except KeyboardInterrupt:
        print("\nWeb server stopped by user.")

if __name__ == '__main__':
    main()