from pydantic import BaseModel, Field
from typing import Literal

class NotificationDTO(BaseModel):
    title: str = Field(description="description of the notification", default="Hackathon 2 esta si que si: Matias Ovalle")
    message: str = Field(description="Message of the notification")
    app_name: str = Field(description="Name of the app that generated the notification", default="WhatsApp")
    timestamp: str = Field(description="Timestamp of the notification", default="1732369225.6608863")
    action_url: str = Field(description="URL to open when the notification is clicked", default="")
    priority: Literal["low", "medium", "high"] = Field(description="Priority of the notification", default="medium")

