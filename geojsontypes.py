from pydantic import BaseModel

class Geometry(BaseModel):
  type: str
  coordinates: list[float]

class Feature(BaseModel):
  type: str
  properties: dict
  geometry: Geometry

class FeatureCollection(BaseModel):
  type: str
  features: list[Feature]
