from fastapi import FastAPI

app = FastAPI(title="FinTrust ML Ops API")

@app.get("/health")
def health_check():
    return {"status": "ok"}