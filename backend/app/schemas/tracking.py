# backend/app/schemas/tracking.py

from pydantic import BaseModel, HttpUrl

class YouTubeUrlPayload(BaseModel):
    url: HttpUrl