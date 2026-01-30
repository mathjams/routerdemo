"""Entry point for running the FastAPI server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        timeout_keep_alive=600,  # 10 minutes for large batches
        limit_concurrency=1000,  # Allow many concurrent requests
        limit_max_requests=10000  # Max requests before worker restart
    )
