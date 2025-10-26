# deep_analysis_agent.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import logging

async def main():
    llm = ChatOpenAI(model="gpt-4o-mini", )  # any reasoning LLM
    server_params = StdioServerParameters(command="python", args=["./server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)         
            agent = create_agent(llm, tools)
            chosen_tool = choose_tool(llm)
            # TODO: get data from Web tool
            result = await agent.ainvoke({"messages": "Compare the evolution of carbon policy trends across documents"})
            print(result)

def choose_tool(prompt: str):
    mapping = {
        "comparative": ["compare", "difference", "versus"],
        "temporal": ["trend", "over time", "pattern"],
        "causal": ["influence", "impact", "cause", "why"],
        "statistical": ["distribution", "correlation", "variance"]
    }
    for key, words in mapping.items():
        if any(w in prompt.lower() for w in words):
            return key
    return "comparative"  # default fallback

if __name__ == "__main__":
    asyncio.run(main())
