from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
from app.agents.resume_agent import run_agent
from app.utils.env import GEMINI_API_KEY

app = FastAPI()

@app.get("/")
def read_root():
    # Serve the index.html file
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "index.html"), media_type="text/html")

class AgentRequest(BaseModel):
    query: str
    thread_id: str = "default"

@app.post("/agent")
def agent_endpoint(payload: AgentRequest):
    try:
        response = run_agent(payload.query, payload.thread_id)
        print("Agent response:", response)  # Debug print
        return {"result": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"Error: {str(e)}"})
