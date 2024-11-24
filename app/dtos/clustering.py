from pydantic import BaseModel, Field
from time import time

class ClusteringDTO(BaseModel):
    timestamp: float = Field(description="Timestamp of the notification", default_factory=time)

