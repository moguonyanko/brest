from pydantic import BaseModel

#Pointかそれ以外かでcoordinatesの入れ子の数が変わる。これを型の違いで表現して
#Geometryのcoordinatesの型に指定したい。
class PointCoordinates(BaseModel):
  coords: list[list[float]]

  def to_geojson(self):
    return self.coords

class NotPointCoordinates(BaseModel):
  coords: list[list[list[float]]]

  def to_geojson(self):
    return self.coords

class Geometry(BaseModel):
  type: str
  # coordinates: PointCoordinates | NotPointCoordinates
  coordinates: list

  def to_geojson(self):
    return {"type":self.type, "coordinates":self.coordinates}

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
