from fastapi import FastAPI, HTTPException, status, Body, Depends, Response
from fastapi_mcp import FastApiMCP

"""
## 参考
* https://www.infoq.com/news/2025/04/fastapi-mcp/
* https://github.com/tadata-org/fastapi_mcp?tab=readme-ov-file
"""

app = FastAPI(
    title="Practice MCP",
    description="FastAPI-MCPを調査するためのAPIを提供します。",
    summary="Practice Fasp-MCP API",
    version="0.0.1"
)

mcp = FastApiMCP(
    app,
    name="Practice Fasp-MCP API",
    description="FastAPI-MCPを調査するためのAPIを提供します。",
    describe_full_response_schema=True,
    describe_all_responses=True
)
mcp.mount()

@app.get("/new/endpoint/", tags=["MCP Sample"], operation_id="new_endpoint", 
         response_model=dict[str, str])
async def new_endpoint():
    """
    This endpoint will not be registered as a tool, 
    since it was added after the MCP instance was created
    """
    return {"message": "Hello, world!"}

# But if you re-run the setup, the new endpoints will now be exposed.
mcp.setup_server()


