# main.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from functools import lru_cache
import asyncio

from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters

# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI(
    title="Deep Analysis Agent",
    description="FastAPI + LangChain + FastMCP analytical backend",
    version="1.0.0"
)

# -------------------------------
# Pydantic Schemas
# -------------------------------
class AnalysisRequest(BaseModel):
    prompt: str
    metadata: Dict[str, Any] = {}

class AnalysisResponse(BaseModel):
    result: str

# -------------------------------
# Agent Initialization
# (cached to avoid reinitializing per request)
# -------------------------------
@lru_cache(maxsize=1)
def get_agent_executor():
    async def initialize():
        llm = ChatOpenAI(model="gpt-4o-mini")

        # Path to MCP server file
        server_params = StdioServerParameters(
            command="python",
            args=["./deep_analysis_server.py"]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(llm, tools)
                return agent

    return asyncio.run(initialize())

# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# -------------------------------
# Deep Analysis Endpoint
# -------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    try:
        agent = get_agent_executor()
        result = await agent.ainvoke({"messages": request.prompt})
        return AnalysisResponse(result=str(result))
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
