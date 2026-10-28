import uvicorn

if __name__ == "__main__":
    # One coordinator owns the graph/checkpointer. Do not enable multiple Uvicorn workers.
    uvicorn.run("cpr_agent.api:app", host="127.0.0.1", port=8000, workers=1)
