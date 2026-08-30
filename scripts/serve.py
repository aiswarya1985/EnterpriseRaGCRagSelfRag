import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        app_dir="/app", # or str(ROOT_DIR)
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_config=None,
    )