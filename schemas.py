'''
参考:
https://fastapi.tiangolo.com/tutorial/sql-databases/#create-the-database-models
'''
from pydantic import BaseModel

'''
modelsとschemasを分けている意図は理解できるがmodelsだけにできないものか？
APIで返すオブジェクトのクラスはBaseModelを継承しないといけないのでBaseを継承する必要のある
modelsと同じにはできないか。。。
'''


class ItemBase(BaseModel):
    title: str
    description: str | None = None


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str

'''
modelsのUserと違いこちらにはパスワードは含まれない。schemasにはAPIから返される要素を定義する。
modelsにはデータベースのテーブルと対応する要素を定義する。
'''
class User(UserBase):
    id: int
    is_active: bool
    items: list[Item] = []

    class Config:
        orm_mode = True
