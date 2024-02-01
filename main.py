'''
参考:
https://fastapi.tiangolo.com/ja/tutorial/
'''
from fastapi import FastAPI, Query, Path, Body, Cookie, Header, Response, status, BackgroundTasks
from fastapi import Form, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from starlette.exceptions import HTTPException as StarletteHTTPException
from enum import Enum
import json
from typing import Union, Annotated, Any
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from uuid import UUID
from datetime import datetime, time, timedelta, timezone
import os
import time as processtime #datetimeのtimeオブジェクトと衝突するので別名を付ける。
#TODO: エラーになるので一旦コメントアウトする。
# from sqlalchemy.orm import Session
# from . import crud, models, schemas
# from .database import SessionLocal, engine

def report_brest_service():
    print(f"REQUEST:{datetime.now()}")

tags_metadata = [
    {
        "name": "users",
        "description": "サンプルアプリのユーザー",
    },
    {
        "name": "items",
        "description": "サンプルアプリのアイテム",
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
    }
]

brest_service = FastAPI(
    title="FastAPI Example API",
    description="FastAPIの使い方を学ぶためのサンプルAPIをまとめています。",
    summary="FastAPIサンプル集",
    version="0.9.0",
    terms_of_service="https://localhost/brest/",
    openapi_tags=tags_metadata,
    dependencies=[Depends(report_brest_service)])

#staticディレクトリ以下に静的ファイルを配置する。ディレクトリが存在しないとエラーになる。
brest_service.mount("/static", StaticFiles(directory="static"), name="static")

'''
ここに含まれないオリジンからのリクエストに対するレスポンスヘッダにはaccess-control-allow-originが付与されない。
'''
origins = [
    "https://localhost",
    "http://localhost:8080",
    "https://myhost",
    "http://myhost:8000"
]

brest_service.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"] #本来は全てのヘッダを認めるべきではない。
)

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
    price: float = Field(ge=0.0, description="品物の値段です。", examples=[1.0])
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
@brest_service.get(APP_ROOT + "items/", tags=["items"])
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
@brest_service.get(APP_ROOT + "groups/{group_name}/users/{user_name}", tags=["users"])
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

'''
Response系以外を戻り値に指定したければresponse_model=Noneを指定しておく。
'''
@brest_service.get(APP_ROOT + "search_dict/", response_model=None)
async def get_search_word_dict(test_mode: bool = False, q: str = "") -> Response | dict:
    if test_mode:
        return JSONResponse({"query": q})
    search_url = f"https://www.google.co.jp/search?q={q}"
    return {"search_url": search_url}

sample_response_items = {
    "1": {
        "item_name": "Apple",
        "description": "赤いりんごです",
        "price": 200
    },
    "2": {
        "item_name": "Orange",
        "description": "橙みかんです",
        "price": 130
    },
    "3": {
        "item_name": "Melon",
        "description": "メロンは高い",
        "price": 2000
    }
}

'''
response_model_exclude_unset=Falseだとクラス側で定義したデフォルト値を使ってオブジェクトの生成が行われる。
'''
@brest_service.get(APP_ROOT + "response_items/{item_id}", 
                   response_model=MyItem, response_model_exclude_unset=False)
async def get_response_item(item_id: str):
    if item_id in sample_response_items:
        return sample_response_items.get(item_id)
    else:
        return {"item_name": "Empty", "price": 0}

'''
レスポンスにはresponse_model_includeに指定したプロパティしか含まれなくなる。
'''
@brest_service.get(APP_ROOT + "response_items_price_and_tax/{item_id}", 
                   response_model=MyItem, 
                   #setではなくlistやtupleで指定してもsetに自動的に変換される。すなわち重複した値は1つにまとめられる。
                   #response_model_include=["item_name", "price", "tax", "item_name"],
                   response_model_include={"item_name", "price", "tax"})
async def get_response_item_nodesc(item_id: str):
    if item_id in sample_response_items:
        return sample_response_items.get(item_id)
    else:
        return {"item_name": "Empty"}

'''
レスポンスにはresponse_model_excludeに指定したプロパティ以外が含まれる。
'''
@brest_service.get(APP_ROOT + "response_items_exclude_desc/{item_id}", 
                   response_model=MyItem, 
                   response_model_exclude={"description"})
async def get_response_item_nodesc(item_id: str):
    if item_id in sample_response_items:
        return sample_response_items.get(item_id)
    else:
        return {"item_name": "Empty"}

#TODO: 継承ではなく委譲で共通部分を集約したい。
# class MyCardBasicInfo(BaseModel):
#     email: EmailStr
#     description: str | None = None    

class MyCardIn(BaseModel):
    email: EmailStr
    description: str | None = None    
    password: str

class MyCardInDB(BaseModel):
    email: EmailStr
    description: str | None = None    
    cipher_password: str

class MyCardOut(BaseModel):
    email: EmailStr
    description: str | None = None    
        
def do_cipher(password: str) -> str:
    return f"SECRET-{password}"

def save_my_card(my_card_in: MyCardIn) -> MyCardInDB:
    my_card_in_db = MyCardInDB(email=my_card_in.email, description=my_card_in.description,
                               cipher_password=do_cipher(my_card_in.password))
    print("カード情報を保存したとします。")
    return my_card_in_db

@brest_service.post(APP_ROOT + "mycard/", response_model=MyCardOut)
async def register_my_card(my_card_in: MyCardIn):
    my_card_in_db = save_my_card(my_card_in)
    #BaseModelを継承していないとmodel_dumpできない。
    #my_card_in_dbをそのまま返してもMyCardOutで見える範囲のプロパティしか公開されないが
    #それでもパスワードを含むMyCardInDBオブジェクトをそのまま返すのは抵抗がある。
    #response_modelさえ正確に指定していれば誤ったオブジェクトを返すリスクが下がるともいえる。
    return my_card_in_db
    #return MyCardOut(**my_card_in_db.model_dump())

class SampleModel(BaseModel):
    name: str

class NumberModel(SampleModel):
    number: int

class UserModel(SampleModel):
    age: int
    favorites: list[str] = []

sample_models = {
    1: {
        "name": "test model",
        "number": 1
    },
    2: {
        "name": "Joe",
        "age": 32,
        "favorites": ["Apple", "Orange"]
    }
}

@brest_service.get(APP_ROOT + "samplemodel/{samplemodel_id}", 
                   response_model=NumberModel | UserModel)
async def get_sample_model(samplemodel_id: str):
    return sample_models[int(samplemodel_id)]

@brest_service.get(APP_ROOT + "allsamplemodels/", 
                   response_model=list[SampleModel])
async def get_all_sample_models():
    return sample_models.values()

'''
SampleModel型で参照できるフィールドしかレスポンスには含まれない。
'''
@brest_service.get(APP_ROOT + "samplemodeldict/", 
                   response_model=dict[int, SampleModel])
async def get_sample_model_dict():
    return sample_models

'''
リクエストパラメータで受け取りたければPathではなくQueryでパラメータを指定する必要がある。
'''
@brest_service.post(APP_ROOT + "sampleaccount/", response_model=dict[str, EmailStr], 
                    status_code=status.HTTP_201_CREATED)
async def echo_sample_account(email: EmailStr = Query(example="sample@mymail.co.jp")):
    return {"email": email}

@brest_service.post(APP_ROOT + "auth/", response_model=dict[str, str])
async def echo_authentication(username: Annotated[str, Form(example=["テストユーザー"])], 
                              password: Annotated[str, Form(example=["8文字以上のパスワード"])]):
    if len(password) <= 8:
        return {"status": "400"}    
    return {"username": username, "password": "*****", "status": "200"}

@brest_service.post(APP_ROOT + "files/", status_code=status.HTTP_201_CREATED, 
                    response_model=dict[str, str])
async def upload_tmp_json_file(file: Annotated[UploadFile, File(description="アップロードされたファイルを読み込みます")]):
    data = { "content": (await file.read()).decode('utf-8') }
    path = "tmp/uploadfile.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

@brest_service.post(APP_ROOT + "filesize/", response_model=dict[str, int])
async def calc_file_size(file: Annotated[bytes, File(description="ファイルサイズが何バイトか計算します")]):
    return {"file_size": len(file)}

@brest_service.post(APP_ROOT + "allfilenames/", response_model=dict[str, list[str]])
async def get_all_file_names(files: Annotated[list[UploadFile], File(description="アップロードされたファイル群の名前一覧を返します")]):
    return {"file_names": [file.filename for file in files]}

@brest_service.post(APP_ROOT + "sumfiles/", response_model=dict[str, int])
async def sum_upload_files(files: Annotated[list[bytes], File(description="ファイル群のサイズ合計を計算します")]):
    return {"sum": sum([len(file) for file in files])}

@brest_service.post(APP_ROOT + "filemetadata/", response_model=dict[str, Any])
async def get_file_metadata(upload_file: Annotated[UploadFile, File()],
                            #オプションパラメータにするとファイル用のUIがドキュメントに表示されない。
                            sample_file: Annotated[bytes | None, File()] = None,
                            #intとして不適切な値を入力された場合はエラーにできる。
                            sample_number: Annotated[int | None, Form()] = None):
    return {
        "uploadFileContentType": upload_file.content_type, 
        "uploadFileSize": upload_file.size,
        "sampleFileSize": len(sample_file) if sample_file else 0,
        #float('nan')を指定するとJSONのレスポンスを生成するタイミングでエラーになる。
        "sampleNumber": sample_number if sample_number else -1 
    }

sample_records: dict[str, str] = {
    "A001": "Hello",
    "B001": "こんにちは",
    "C001": "Bonjour"
}

#文字列を単体で返してもJSONとしては適切と解釈される。
@brest_service.get(APP_ROOT + "samplerecord/{record_id}", response_model=str)
async def get_sample_record(record_id: str):
    if not record_id in sample_records:
        raise HTTPException(status_code=404, 
                            detail=f"{record_id}に対応するレコードは存在しません",
                            headers={
                                "X-HasError": True
                            })
    return sample_records.get(record_id)

class MySampleResourceException(Exception):
    def __init__(self, user_name: str, *args: object) -> None:
        super().__init__(*args)
        self.user_name = user_name

@brest_service.exception_handler(MySampleResourceException)
async def my_custom_exception_handler(request: Request, ex: MySampleResourceException) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "message": f"{ex.user_name}によるリソースへのアクセスは禁止されています。"
        }
    )

@brest_service.exception_handler(HTTPException)
async def common_http_exception_handler(request: Request, ex: HTTPException):
    return await http_exception_handler(request, ex)
#上とほぼ同じ結果を返す。
#    return PlainTextResponse(ex.detail, status_code=ex.status_code)

@brest_service.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, err: RequestValidationError):
    return await request_validation_exception_handler(request, err)
#上とほぼ同じ結果を返す。
    # return JSONResponse(content=jsonable_encoder({
    #                         "detail": err.errors(),
    #                         "body": err.body
    #                     }), 
    #                     status_code=status.HTTP_400_BAD_REQUEST)

class MyTags(Enum):
    validators = "validators"
    authenticators = "authenticators"

@brest_service.get(APP_ROOT + "sampleresource/{user_name}", response_model=dict[str, str],
                              tags=[MyTags.validators, MyTags.authenticators])
async def get_sample_resource(user_name: Annotated[str, Path(example="admin")], 
                              age: Annotated[int, Query(example=18)],
                              message: Annotated[str | None, Query(example="test")]):
    if not user_name == "admin":
        raise MySampleResourceException(user_name=user_name)
    if not age >= 18:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Adult only")
    if len(message) <= 0:
        raise RequestValidationError(errors=["Bad empty message"])
    return {"user_name": user_name}

@brest_service.post(APP_ROOT + "checkmyprofile/", response_model=MyProfile, 
                    tags=[MyTags.validators],
                    summary="Check my profile",
                    description="Check MyProfile Object",
                    response_description="Valid MyProfile Object as JSON")
async def check_my_profile(profile: MyProfile): 
    if len(profile.name) <= 0:
        raise RequestValidationError(errors=["名前を入力してください"])
    return profile

class MyClock(BaseModel):
    name: str
    current: datetime
    degital: bool = False

#JSON形式しか受け付けない変数という設定
json_my_clock = {
    "name": "default",
    "current": datetime.now,
    "degital": False
}
    
@brest_service.put(APP_ROOT + "currenttime/", response_model=dict[str, Any])
async def update_clock(clock: MyClock):
    json_clock = jsonable_encoder(clock)
    json_my_clock = json_clock
    return json_my_clock

class MyCharacter(BaseModel):
    name: str = ""
    description: str | None = None
    power: int = 0
    tags: list[str] = []

my_characters = {
    "taro": {
        "name" : "Taro",
        "power" : 100,
        "tags": ["main", "test"]
    },
    "joe": {
        "name" : "Joe",
        "description": "Mecanic",
        "power" : 80
    },
    "usao": {
        "name" : "Usao",
        "power" : 5,
        "tags": ["モブ", "古参"]
    } 
}

#丸々データを置き換えるのでPUTを使う。
@brest_service.put(APP_ROOT + "mycharacters/{character_id}", response_model=MyCharacter)
async def update_my_character(character_id: str, my_character: MyCharacter):
    character_data = jsonable_encoder(my_character)
    my_characters[character_id] = character_data
    return character_data

@brest_service.patch(APP_ROOT + "mycharacters/{character_id}", response_model=MyCharacter)
async def patrial_update_my_character(character_id: str, my_character: MyCharacter):
    current_data = my_characters[character_id]
    current_character = MyCharacter(**current_data)
    new_data = my_character.model_dump(exclude_unset=True)
    updated_character = current_character.model_copy(update=new_data)
    my_characters[character_id] = jsonable_encoder(updated_character)
    return updated_character

sample_config = {
    "client": {
        "description": "test client config"
    },
    "server": {
        "description": "test server config"
    }
}

async def load_common_config(config_id: str | None = None):
    if not config_id:
        return {}
    return sample_config[config_id]

common_config = Annotated[dict, Depends(load_common_config)]

@brest_service.get(APP_ROOT + "clienttestapp/", response_model=dict[str, Any])
async def get_client_test_app_config(config: common_config):
    return config

@brest_service.get(APP_ROOT + "servertestapp/", response_model=dict[str, Any])
async def get_server_test_app_config(config: common_config):
    return config

fake_user_db = [
    {
        "name": "Mike"
    },
    {
        "name": "Taro"
    },
    {
        "name": "Joe"
    },
]

class CommonQueryParam:
    def __init__(self, query: str = "", skip: int = 0, limit: int = len(fake_user_db)):
        self.query = query
        self.skip = skip
        self.limit = limit

@brest_service.get(APP_ROOT + "userdata/", response_model=dict[str, Any])
#async def get_fake_user_data(param: Annotated[CommonQueryParam, Depends(CommonQueryParam)]):
#重複する型名（ここではCommonQueryParam）は省略可能。
async def get_fake_user_data(param: Annotated[CommonQueryParam, Depends()]):
    userdata = {}
    if param.query:
        userdata.update({"query": param.query})
    userdata_list = fake_user_db[param.skip : param.skip + param.limit]
    userdata.update({ "userDataList": userdata_list })
    return userdata

def query_checker(query: str) -> str:
    #Dummy check
    return query

def get_test_query(query: Annotated[str, Depends(query_checker, use_cache=False)],
                   last_query: Annotated[str, Cookie()] = "") -> str:
    return query if query else last_query

@brest_service.get(APP_ROOT + "testquesy/", response_model=dict[str, str])
async def read_test_query(query: Annotated[str, Depends(get_test_query, use_cache=False)]):
    return {"query": query}

async def check_token(x_token: Annotated[str, Header()]):
    if x_token == "secret":
        return x_token
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail="トークンが不正です")

async def check_origin(origin: Annotated[str, Header()]):
    if origin == "https://localhost" or origin == "http://127.0.0.1:8000":
        return origin
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail="オリジンが不正です")

#postでないとOriginヘッダーを送信できない。
@brest_service.post(APP_ROOT + "member/", dependencies=[Depends(check_token), 
                                                       Depends(check_origin)],
                                                       response_model=dict[str, str])
async def get_valid_member():
    return {"username": "validuser"}

class TestAdminUser(BaseModel):
    name: str
    description: str

sample_user_database = {
    "admin": {
        "name": "Mike",
        "description": "Test administrator" 
    }
}

class AdminError(Exception):
    pass

class TestAdminUserLoader():
    def load(self) -> str:
        return "Mike"

    def close(self) -> bool:
        return True

class MyAdminContextManager:
    def __init__(self):
        self.loader = TestAdminUserLoader()

    def __enter__(self):
        return self.loader.load()
                  
    def __exit__(self, exc_type, exc_value, traceback):
        return self.loader.close()

def get_test_admin_username() -> str:
    with MyAdminContextManager() as manager:
        yield manager
    
@brest_service.get(APP_ROOT + "testadminuser/{user_id}", response_model=TestAdminUser)
#user_idはクエリパラメータではなくパス。そのためQueryではなくPathを使って型定義する。
async def get_test_admin_user(user_id: Annotated[str, Path(example="admin")], 
                              username: Annotated[str, Depends(get_test_admin_username)]):
    if user_id not in sample_user_database:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Not found user_id:{user_id}")
    user = sample_user_database[user_id]
    if user["name"] != username:
        raise AdminError(username)
    return user 

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@brest_service.post(APP_ROOT + "testtoken/", response_model=dict[str, str])
async def create_test_token(token: Annotated[str, Depends(oauth2_scheme)]):
    return {"token": token}

fake_auth_user_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}

class MyToken(BaseModel):
    access_token: str
    token_type: str

class MyTokenData(BaseModel):
    username: str = ""

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password) -> bool:
    return password_context.verify(plain_password, hashed_password)

def get_password_hash(password) -> str:
    return password_context.hash(password)

def fake_hashed_password(password: str) -> str:
    return f"fakehashed_{password}"

class AuthUser(BaseModel):
    username: str = ""
    full_name: str = ""
    email: EmailStr | None = None
    disabled: bool = False
    invalid: bool = False

#こういう継承はあまり好ましくない。
class AuthUserInDB(AuthUser):
    hashed_password: str

def lookup_auth_user(db, token: str) -> AuthUser:
    if token in db:
        user_dict = db[token]
        return AuthUserInDB(**user_dict)
    return AuthUserInDB(invalid=True)

def fake_decode_user(token) -> AuthUser:
    return lookup_auth_user(fake_auth_user_db, token)

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> AuthUser:
    ex = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="不正なユーザーです",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
        username = payload.get("sub")
        if not username:
            raise ex
        token_data = MyTokenData(username=username)
    except JWTError:
        raise ex
    user = lookup_auth_user(fake_auth_user_db, token_data.username)
    if user.invalid:
        raise ex
    return user

async def get_current_active_user(current_user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUser:
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="無効なユーザーです")
    return current_user

def authenticate_user(db: dict[str, Any], username: str, password: str) -> bool:
    user = lookup_auth_user(db, username)
    if not user:
        return False
    elif not verify_password(password, user.hashed_password):
        return False
    else:
        return True

def create_access_token(data: dict[Any, Any], expire_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expire_delta:
        expire = datetime.now(timezone.utc) + expire_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@brest_service.get(APP_ROOT + "authusers/me", response_model=AuthUser)
async def read_authuser_me(current_user: Annotated[AuthUser, Depends(get_current_active_user)]):
    return current_user

@brest_service.post(APP_ROOT + "token", response_model=MyToken)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(fake_auth_user_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="認証エラーです")
    access_token_expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data = {"sub": user.username}, expire_delta=access_token_expire
    )
    return MyToken(access_token=access_token, token_type="bearer")

@brest_service.get(APP_ROOT + "authusers/me/items/", response_model=list[dict[str, str]])
async def get_authuser_items(current_user: Annotated[AuthUser, Depends(get_current_active_user)]):
    return [
        {
            "name": "Banana",
            "owner": current_user.username
        }
    ]

@brest_service.middleware("http")
async def set_process_time_header(request: Request, next_func):
    start_time = processtime.time()
    response = await next_func(request)
    process_time = processtime.time() - start_time
    response.headers["X-Process-Time"] = str(process_time) #文字列でないとエラー
    return response

'''
参考:
https://fastapi.tiangolo.com/tutorial/sql-databases/#main-fastapi-app
'''
# models.Base.metadata.create_all(bind=engine)

# # Dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# @brest_service.post(APP_ROOT + "dbusers/", response_model=schemas.User)
# def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
#     db_user = crud.get_user_by_email(db, email=user.email)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Email already registered")
#     return crud.create_user(db=db, user=user)


# @brest_service.get(APP_ROOT + "dbusers/", response_model=list[schemas.User])
# def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     users = crud.get_users(db, skip=skip, limit=limit)
#     return users


# @brest_service.get(APP_ROOT + "dbusers/{user_id}", response_model=schemas.User)
# def read_user(user_id: int, db: Session = Depends(get_db)):
#     db_user = crud.get_user(db, user_id=user_id)
#     if db_user is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return db_user


# @brest_service.post(APP_ROOT + "dbusers/{user_id}/items/", response_model=schemas.Item)
# def create_item_for_user(
#     user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)
# ):
#     return crud.create_user_item(db=db, item=item, user_id=user_id)


# @brest_service.get(APP_ROOT + "dbitems/", response_model=list[schemas.Item])
# def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     items = crud.get_items(db, skip=skip, limit=limit)
#     return items

def print_log(text: str):
    print(f"MYLOG:{text}")

'''
middlewareでも同じことはできそうだがログ出力などはBackgroundTasksの方がふさわしそうである。
'''
@brest_service.post(APP_ROOT + "testlog/")
async def get_backgroud_sample(text: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(print_log, text=text)
    return {"test": "OK"}

def get_sample_id(background_tasks: BackgroundTasks, sample_id: str):
    if sample_id: 
        background_tasks.add_task(print_log, text=sample_id)
    else: 
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail="sample_idは必須です。")
    return sample_id

@brest_service.post(APP_ROOT + "testlogwithdependency/{sample_id}")
async def get_backgroud_sample(text: str, background_tasks: BackgroundTasks, 
                               sample_id: Annotated[str, Depends(get_sample_id)]):
    background_tasks.add_task(print_log, text=text)
    return {"sampleId": sample_id}
