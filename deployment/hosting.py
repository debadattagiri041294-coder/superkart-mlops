"""
hosting.py
----------
Pushes the deployment files (app.py, Dockerfile, requirements.txt) to a
Hugging Face Space so the Streamlit app goes live / redeploys.

Usage:
    python deployment/hosting.py

Environment variables required:
    HF_TOKEN        - Hugging Face access token with write permission
    HF_SPACE_REPO   - e.g. "your-username/superkart-sales-forecast"
"""

import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_SPACE_REPO = os.environ.get("HF_SPACE_REPO", "your-username/superkart-sales-forecast")
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))

FILES_TO_PUSH = ["app.py",  "requirements.txt"]


def main():
    api = HfApi(token=HF_TOKEN)
    api.create_repo(
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    for filename in FILES_TO_PUSH:
        local_path = os.path.join(DEPLOY_DIR, filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=HF_SPACE_REPO,
            repo_type="space",
        )
        print(f"Pushed {filename} -> {HF_SPACE_REPO}")

    print("Space deployment files pushed. HF Space will rebuild automatically.")


if __name__ == "__main__":
    main()
