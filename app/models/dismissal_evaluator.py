from pydantic import BaseModel, Field

class NotificationDismissal(BaseModel):
    is_dismissible: bool = Field(
        description="Whether the notification is dismissible or not",
    )
    confidence: float = Field(
        description="How confidence you are  that the notification is dismissible"
        "The higher the number, the more certain the model is"
        ", the values are between 0 and 1",
    )
    reason: str = Field(
        description="Why the notification was dismissed or not",
    )

