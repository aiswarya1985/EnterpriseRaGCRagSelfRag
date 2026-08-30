import sys
from pathlib import Path
import uvicorn

# Dynamically find the project root (the directory containing the 'app/' folder)
ROOT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        app_dir=str(ROOT_DIR),  # Points to parent directory of app/
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_config=None,
    )