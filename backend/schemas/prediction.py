from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude in decimal degrees",
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude in decimal degrees",
    )