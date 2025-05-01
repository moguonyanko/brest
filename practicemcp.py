from typing import Annotated, Any
from fastapi import FastAPI, HTTPException, status, Body, Depends, Response
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field, HttpUrl, EmailStr

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

# But if you re-run the setup, the new endpoints will now be exposed.
mcp.setup_server()

APP_BASE = "practicemcp"

@app.get(f"/{APP_BASE}/helloworld/", tags=["MCP Sample"], operation_id="hello_world", 
         response_model=dict[str, str])
async def hello_world():
    """
    This endpoint will not be registered as a tool, 
    since it was added after the MCP instance was created
    """
    return {"message": "Hello, world!"}

@app.get(f"/{APP_BASE}/id/list/", tags=["MCP Sample"], operation_id="get_sample_id_list", 
         response_model=list[str],
         description='サンプルID一覧を返します。')
async def get_sample_id_list():
    return [
        'A001', 'B002', 'C003'
    ]

class Parameters(BaseModel):
    x: int = Field(
        title="x",
        description="第1パラメータ",
        example=1
    )
    y: int = Field(
        title="y",
        description="第2パラメータ",
        example=2
    )

@app.post(f"/{APP_BASE}/addnumbers/", tags=["MCP Sample"], operation_id="add_numbers", 
         response_model=int,
         description='引数を加算します。xとyは必須です。')
async def add_numbers(params: Annotated[Parameters, 
                                    Body(
                        examples=[ 
                            Parameters(
                                x=1,
                                y=2
                            )
                        ]
                    )]):
    """
    引数を加算して返します。
    """
    x = params.x
    y = params.y
    if x is None or y is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="x and y are required")
    
    return x + y

