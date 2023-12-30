from fastapi import FastAPI

brest_app = FastAPI()

@brest_app.get("/")
async def root():
    return "Hello 𩸽を𠮟る𠮷野家〜髙﨑〜彁"

@brest_app.get("/brest/")
async def brest_root():
    return {"message": "Brest Root"}
