from pydantic import BaseModel, Field

class Notification(BaseModel):
    message: str = Field(
        description="The summarized message of the notification",
    )
    category: str = Field(
        description="The category of the notification",
    )

