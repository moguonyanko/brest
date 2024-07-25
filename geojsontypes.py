from pydantic import BaseModel

class Geometry(BaseModel):
  type: str
  coordinates: list[list[list[float]]]

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
