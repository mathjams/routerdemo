"""Start the API server:  python backend/run.py   (listens on $PORT, default 8000)"""
import os
import sys

import uvicorn

# Make the `backend` package importable no matter which directory this is started from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
