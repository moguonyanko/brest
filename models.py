'''
参考:
https://fastapi.tiangolo.com/tutorial/sql-databases/#create-the-database-models
'''
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base

'''
もしテーブルの定義を変えたら以下のクラス群の内容も修正が必要になると思われる。
ORMによりデータベースをプログラム内で扱う負担は減るが変更には弱くなるということ。
ORMを使っていなくてもカラム名を変えたりしたら影響は発生しうるが、以下の仕組みでは
実際の処理では使用していないカラムやテーブル間の関係性に対する変更の影響まで発生してしまう。
'''

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="items")
  