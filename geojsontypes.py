from pydantic import BaseModel

class Coordinates(BaseModel):
  def to_geojson(self):
    return self.coords

#Pointかそれ以外かでcoordinatesの入れ子の数が変わる。これを型の違いで表現して
#Geometryのcoordinatesの型に指定したいが422エラーが発生する。
#リクエストボディのJSON内の配列に対してBaseModelの方を適用させることはできないのかもしれない。
class PointCoordinates(Coordinates):
  coords: list[float]

class NotPointCoordinates(Coordinates):
  coords: list[list[list[float]]]

class Geometry(BaseModel):
  type: str
  #以下だと422エラーになってしまう。
  # coordinates: PointCoordinates | NotPointCoordinates
  coordinates: list[float] | list[list[list[float]]]
  # coordinates: list #これも動作するが型が曖昧すぎる。

  def to_geojson(self):
    return {"type":self.type, "coordinates":self.coordinates.to_json()}

class Feature(BaseModel):
  type: str
  properties: dict | None
  geometry: Geometry

  def to_geojson(self):
    return {"type":self.type, "properties":self.properties, "geometry": self.geometry.to_json()}

class FeatureCollection(BaseModel):
  type: str
  features: list[Feature]

  def to_geojson(self):
    return {"type":self.type, "features":[feature.to_json() for feature in self.features]}
