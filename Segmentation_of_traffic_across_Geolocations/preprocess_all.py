import subprocess

scripts = [
    "preprocess_bdd.py",
    "preprocess_city.py",
    "preprocess_idd.py"
]

for script in scripts:
    print(f"\n=== Running {script} ===")
    result = subprocess.run(["python", script])
    
    if result.returncode != 0:
        print(f"⚠️ {script} exited with an error (code {result.returncode}). Stopping execution.")
        break
    else:
        print(f"✅ Finished {script} successfully.")