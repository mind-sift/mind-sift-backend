from pydantic import BaseModel, Field
from uuid import uuid4

class NotificationDTO(BaseModel):
    id: str = Field(description="ID unico", default_factory=lambda: str(uuid4()))
    notification_id: str = Field(description="ID of the notification", default_factory=lambda: str(uuid4()))
    notification_tag: str = Field(description="Tag of the notification", default="weird_string")
    package_name: str = Field(description="Name of the package that generated the notification", default="com.whatsapp")
    source_user_iden: str = Field(description="User ID of the source user", default="123456789")
    title: str = Field(description="description of the notification", default="Hackathon 2 esta si que si: Matias Ovalle")
    message: str = Field(description="Message of the notification")
    app_name: str = Field(description="Name of the app that generated the notification", default="WhatsApp")
    timestamp: float = Field(description="Timestamp of the notification")
    category: str = Field(description="cateogry", default="")

