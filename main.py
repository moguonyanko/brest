from fastapi import FastAPI

my_sample_app = FastAPI()

@my_sample_app.get("/")
async def root():
    return {"message": "𩸽を𠮟る𠮷野家〜髙﨑"}
