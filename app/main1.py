from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
from app.agent import run_agent
from dotenv import load_dotenv

# Google Gemini integration
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None

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
        return {"result": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"Error: {str(e)}"}) 