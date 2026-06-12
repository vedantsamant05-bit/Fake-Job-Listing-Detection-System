import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def download_emscad():
    """Download the EMSCAD dataset from Kaggle."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / "fake_job_postings.csv"
    if csv_path.exists():
        print(f"Dataset already exists at {csv_path}")
        return

    # Requires KAGGLE_USERNAME and KAGGLE_KEY in environment or ~/.kaggle/kaggle.json
    os.system(
        f"kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction "
        f"-p {data_dir} --unzip"
    )
    print(f"Dataset downloaded to {data_dir}")

if __name__ == "__main__":
    download_emscad()