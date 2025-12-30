from fastapi import FastAPI

app = FastAPI(
    title="GuitarVision API",
    description="Backend service for GuitarVision",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify service status.
    """
    return {"status": "ok"}
