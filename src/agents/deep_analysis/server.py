# deep_analysis_server.py
from mcp.server.fastmcp import FastMCP
from system_prompts import SYSTEM_PROMPTS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

mcp = FastMCP("DeepAnalysisTools")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@mcp.tool()
def comparative_analysis(payload: dict) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPTS["comparative_analysis"]),
        HumanMessage(content=payload["text"])
    ]
    response = llm.invoke(messages)
    return {"result": response.content}

@mcp.tool()
def temporal_pattern_analysis(payload: dict) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPTS["temporal_pattern_analysis"]),
        HumanMessage(content=payload["text"])
    ]
    response = llm.invoke(messages)
    return {"result": response.content}

@mcp.tool()
def causal_reasoning(payload: dict) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPTS["causal_reasoning"]),
        HumanMessage(content=payload["text"])
    ]
    response = llm.invoke(messages)
    return {"result": response.content}

@mcp.tool()
def statistical_analysis(payload: dict) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPTS["statistical_analysis"]),
        HumanMessage(content=payload["text"])
    ]
    response = llm.invoke(messages)
    return {"result": response.content}
