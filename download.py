import subprocess

if __name__ == "__main__":
    repo_id = "code-philia/CoEdPilot-extension-beta"

    # execute huggingface-cli download command
    try:
        subprocess.run(["huggingface-cli", "download", repo_id, "--cache-dir", "./models"], check=True)
        print(f"Successfully downloaded {repo_id}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading {repo_id}: {e}")