from fastapi import FastAPI

brest_service = FastAPI()
APP_ROOT = "/brest/"

@brest_service.get("/")
async def root():
    return "Hello FastAPI"

@brest_service.get(APP_ROOT)
async def brest_root():
    return {"message": "Brest Root"}

#関数名が衝突する場合は定義された順に評価される。
@brest_service.get(APP_ROOT + "echo/hello")
async def brest_echo():
    return {"message": 'World'}

@brest_service.get(APP_ROOT + "echo/{message}")
async def brest_echo(message: str):
    return {"message": message}

#関数名が他と重複してもエラーにはならない。しかしドキュメントがややこしくなるので好ましくない。
@brest_service.get(APP_ROOT + "pow/{number}")
async def brest_pow(number: int): #型情報を書くことで整数でない値を渡した時に詳細なエラーを表示できる。
    return {"result": number ** 2}

