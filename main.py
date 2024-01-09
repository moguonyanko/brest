'''
参考:
https://fastapi.tiangolo.com/ja/tutorial/
'''
from fastapi import FastAPI, Query, Path, Body, Cookie, Header, Response
from fastapi.responses import JSONResponse, RedirectResponse
from enum import Enum
import json
from typing import Union, Annotated, Any
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from uuid import UUID
from datetime import datetime, time, timedelta

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

class MyImage(BaseModel):
    url: HttpUrl #妥当なURLかどうかどチェックすることができるようになる。
    file_name: str

class MyItem(BaseModel):
    item_name: str = Field(examples=["no name"])
    description: str | None = Field(default=None, title="品物の説明", max_length=10, examples=["特になし"])
    price: float = Field(gt=0.1, description="品物の値段です。", examples=[1.0])
    tax: float | None = None
    tags: set[str] = set() #setで宣言してもリクエストボディでは配列でパラメータを渡すことになる。
    images: list[MyImage] = []

    option_config: dict = {
        "testcode": "TEST"
    }
    
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
        price = 300,
        tags = {"test", "sample"},
        images = [
            MyImage(url="https://localhost/sampleimages/", file_name="sampleimage1.jpg"),
            MyImage(url="https://myhost/testimages/", file_name="testimage1.png"),
            MyImage(url="https://myhost/testimages/", file_name="testimage2.png")
        ]
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

def get_sample_items():
    return {
        1: {
           "name": "Apple" 
        },
        2: {
           "name": "Banana" 
        },
        3: {
           "name": "Orange" 
        }
    }

@brest_service.get(APP_ROOT + "items/{item_id}")
async def get_item_by_id(
    item_id: Annotated[int, Path(title="品物取得用ID")], #itemsのキーがintの場合
    q: Annotated[str | None, Query(alias="member-query")] = None
    ):
    items = get_sample_items()
    #パスはstrなのでintが必要な場合は変換が必要
    item = items.get(int(item_id)) or {}
    if item and q:
        item.update({"q": q})
    return item

@brest_service.get(APP_ROOT + "items_annotated/{item_id}")
async def get_item_by_id_and_annotated(
    #item_idが1以上10未満でなければエラーになる。
    #Pathはバリデーションのためにあると考えてよさそう。
    item_id: Annotated[int, Path(title="品物取得用ID", ge=1, lt=10)], 
    size: Annotated[float, Query(gt=0, lt=10.1)],
    q: str | None = None):
    items = get_sample_items()
    item = items.get(int(item_id)) or {}
    if item and q:
        item.update({"q": q})
    if size:
        item.update({"size": size})
    return item

'''
引数名を指定しないとエラーになるコンストラクタが宣言されていると考えてよい。
'''
class MyUser(BaseModel):
    name: str
    age: int

sample_my_users: [MyUser] = [
    MyUser(name="Masao", age=38)
]

sample_item_dict = {
    "1": {
        "item": sample_items[0],
        "user": sample_my_users[0]
    }
}

'''
Body()を使って指定することでクエリパラメータではなくリクエストボディであるとFastAPIに認識させる。
'''
@brest_service.put(APP_ROOT + "items/{item_id}")
async def save_item(item_id: int, 
                    item: Annotated[MyItem, 
                                    Body(
                        openapi_examples={
                            "one_example": {
                                "summary": "Example No.1",
                                "description": "例1",
                                "value": {
                                    "item_name": "my sample",
                                    "description": "nothing",
                                    "price": 100,
                                    "tax": 0.1
                                }
                            },
                            "two_example": {
                                "summary": "Example No.2",
                                "description": "例2",
                                "value": {
                                    "item_name": "my test",
                                    "price": 999,
                                }
                            }
                        }
                    )], 
                    user: Annotated[MyUser, Body(embed=True)],
                    memo: Annotated[str, Body()] = "特になし",
                    test_code: Annotated[int, Body(ge=0)] = 0):
    new_item = {
        item_id: {
            "item": item,
            "user": user,
        }
    }
    sample_item_dict.update(new_item)
    sample_item_dict.update({"memo": memo})
    sample_item_dict.update({"test_code": test_code})
    return sample_item_dict

class MyOffer(BaseModel):
    name: str
    description: str = ""
    price: float
    items: list[MyItem] #[MyItem]だけだとエラーになる。

@brest_service.post(APP_ROOT + "offers/")
async def echo_offer(offer: MyOffer):
    return offer

@brest_service.post(APP_ROOT + "images/")
async def echo_images(images: list[MyImage]):
    return images

@brest_service.get(APP_ROOT + "duration/{item_id}")
async def get_duration(
    item_id: UUID,
    start_datetime: Annotated[datetime | None, None] = None,
    end_datetime: Annotated[datetime | None, None] = None,
    repeat_at: Annotated[time | None, None] = None,
    proccess_after: Annotated[timedelta | None, None] = None
):
    start_process = start_datetime + proccess_after
    duration = end_datetime - start_process
    return {
        "item_id": item_id,
        "duration": duration
    }

@brest_service.get(APP_ROOT + "samplecookie/")
async def echo_cookie(sample_id: str | None = Cookie(default=None)):
    return {"sample_id": sample_id}

@brest_service.get(APP_ROOT + "useragent/")
async def echo_useragent(user_agent: Annotated[str | None, Header()] = None):
    return {"UserAgent": user_agent}

@brest_service.get(APP_ROOT + "duplicateheaders/")
async def get_sample_token_headers(x_my_token: Annotated[list[str], Header()] = []):
    #複数のヘッダーを送ってもx_my_tokenが文字列一つで構成されるリストになってしまう。
    return {"xMyToken": x_my_token[0].split(",")}

class MyProfile(BaseModel):
    name: str = Field(default="", examples=["Taro"])
    age: int = Field(ge=0, le=130, examples=[18])
    favorites: list[str] = []
    email: EmailStr | None = Field(default=None, examples=["mymail@dummymail.co.jp"])

class MyProfileInput(MyProfile):
    password: str = Field(examples=["Brest2024_pass"])

sample_my_profiles = [
    MyProfile(name="Mike", password="MiPass8374W", age=43, 
              favorites=["Baseball", "Car", "Walking"])
]

@brest_service.post(APP_ROOT + "myprofile/")
async def save_my_profile(profile: MyProfileInput) -> MyProfile: #passwordが見えない型で返す。
    sample_my_profiles.append(profile)
    return profile

@brest_service.get(APP_ROOT + "myprofile/", response_model=list[MyProfile])
async def get_all_my_profiles() -> list[MyProfile]:
    return sample_my_profiles
    
@brest_service.get(APP_ROOT + "search/")
async def get_search_word(test_mode: bool = False, q: str = "") -> Response:
    if test_mode:
        return JSONResponse({"query": q})
    search_url = f"https://www.google.co.jp/search?q={q}"
    return RedirectResponse(url=search_url)
