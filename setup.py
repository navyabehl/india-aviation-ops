"""
Run this once before launching the dashboard.
Fetches source data, runs ETL pipeline, trains predictive models.
"""
import subprocess
import sys
import os

def run(script):
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 50)
    print("India Aviation Ops — Setup")
    print("=" * 50)

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Download raw data
    import urllib.request
    files = {
        "data/raw/daily.csv":   "https://raw.githubusercontent.com/Vonter/india-aviation-traffic/main/aggregated/daily.csv",
        "data/raw/carrier.csv": "https://raw.githubusercontent.com/Vonter/india-aviation-traffic/main/aggregated/domestic/carrier.csv",
        "data/raw/city.csv":    "https://raw.githubusercontent.com/Vonter/india-aviation-traffic/main/aggregated/domestic/city.csv",
    }
    for path, url in files.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"  {path} already exists, skipping.")

    print("\nRunning ETL pipeline...")
    run("src/etl.py")

    print("\nTraining models...")
    run("src/model.py")

    print("\n✅ Setup complete! Run: streamlit run app.py")
