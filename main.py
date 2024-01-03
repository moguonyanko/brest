'''
参考:
https://fastapi.tiangolo.com/ja/tutorial/
'''
from fastapi import FastAPI, Query
from enum import Enum
import json
from typing import Union, List
from pydantic import BaseModel

brest_service = FastAPI()
APP_ROOT = "/brest/"

@brest_service.get("/")
async def root():
    return "Hello FastAPI"

@brest_service.get(APP_ROOT)
async def brest_root():
    return {"message": "Brest Root"}

# 関数名が衝突する場合は定義された順に評価される。
@brest_service.get(APP_ROOT + "echo/hello")
async def brest_echo():
    return {"message": 'World'}

@brest_service.get(APP_ROOT + "echo/{message}")
async def brest_echo(message: str):
    return {"message": message}

# 関数名が他と重複してもエラーにはならない。しかしドキュメントがややこしくなるので好ましくない。
@brest_service.get(APP_ROOT + "pow/{number}")
async def brest_pow(number: int):  # 型情報を書くことで整数でない値を渡した時に詳細なエラーを表示できる。
    return {"result": number ** 2}

# class ModelName():
#     def __init__(self, name: str) -> None:
#         self.name = name

# class Models(Enum):
#     foonet = ModelName("foo")
#     barnet = ModelName("bar")
#     baznet = ModelName("baz")

# strを継承させないとパスパラメータの値から定義済みのクラス（ここではModels）を得ることができない。
class Models(str, Enum):
    foonet = "foo"
    barnet = "bar"
    baznet = "bar"

@brest_service.get(APP_ROOT + "models/{model}")
async def get_model_description(model: Models):
    if model is Models.foonet:
        return {"name": model.name, "message": "foonet is poor"}
    if model.value == "bar":
        return {"name": model.name, "message": "barnet is good"}
    return {{"name": model.name, "message": "baznet is stupid"}}

@brest_service.get(APP_ROOT + "files/{file_path:path}")
async def read_json(file_path: str):
    with open(f"/{file_path}", encoding="utf-8") as f:
        return json.loads(f.read())

class MyItem(BaseModel):
    item_name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    
    def __str__(self) -> str:
        return json.dumps({
            "item_name": self.item_name,
            "description": self.description,
            "price": self.price,
            "tax": self.tax
        })

sample_items: [MyItem] = [
    MyItem(
        item_name = "𩸽",
        price = 300
    ),
    MyItem(
        item_name = "𠮷野家",
        price = 10000
    ),
    MyItem(
        item_name = "髙﨑〜彁",
        price = 200
    )
]

'''
Unionでオプションパラメータを定義できる。Union[str, None]の場合はstrのパラメータが指定されればそれを使い
パラメータが指定されなければNoneになる。
'''
@brest_service.get(APP_ROOT + "items/")
async def read_item(start: int = 0, end: int = len(sample_items),
                    description: Union[str, None] = None,
                    nameonly: bool = False):
    items = sample_items[start: end]
    if nameonly:
        items = [item["item_name"] for item in items]
    if description:
        items.append({"description": description})
    return items

sample_users = {
    "soumu": {
        "mike": {
            "name": "Mike",
            "score": 90
        },
        "taro": {
            "name": "Taro",
            "score": 95
        }
    },
    "jinji": {
        "joe": {
            "name": "Joe",
            "score": 88
        }
    }
}

# デフォルト値を指定していないパラメータは必須パラメータとなる。
@brest_service.get(APP_ROOT + "groups/{group_name}/users/{user_name}")
async def get_user(group_name: str, user_name: str, score: Union[int, None] = None):
    user = sample_users.get(group_name).get(user_name)
    if user == None or score and user.get("score") < score:
        return {}
    return user

@brest_service.post(APP_ROOT + "items/{item_id}")
async def register_item(item_id: int, item: MyItem, description: Union[str, None] = None):
    items = {"item_id": item_id, **item.model_dump()}
    if item.tax:
        price_with_tax = item.price * (1 + item.tax)
        items.update({"price_with_tax" : price_with_tax})
    if description:
        items.update({"description": description})
    return items

def get_sample_members():
    sample = {
        "members": [{"id": "Mike"}, {"id": "Taro"}]
    }
    return sample

#第1引数が...のQueryは必須パラメータになる。
@brest_service.get(APP_ROOT + "query/")
async def read_query(p: str = Query(..., min_length=1), 
                     q: Union[str, None] = Query(default=None, max_length=10, 
                                                 min_length=3,
                                                 pattern="[A-Za-z]",
                                                 deprecated=True)):
    sample = get_sample_members()
    sample.update({"p": p})
    if q:
        sample.update({"q": q})
    return sample

#Unionではなくlistで宣言しないとdocs上で複数のパラメータを指定できるUIが適用されない。
@brest_service.get(APP_ROOT + "multiquery/")
async def read_multi_query(q: list = Query(default=["Default"], 
                                           title="Multi Query String", 
                                           description="複数のクエリパラメータを受け取るサンプル関数です。",
                                           max_length=10,
                                           alias="member-query")):
    sample = get_sample_members()
    if q:
        sample.update({"q": q})
    return sample
