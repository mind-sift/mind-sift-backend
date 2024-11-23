from operator import itemgetter
from app.chat_models.default import get_model_chain
from app.dtos.notification import NotificationDTO
from app.models.notification_summary import Notification
from langchain_core.runnables import Runnable
from app.prompts.summary import summary_messages_template

def get_summarization_chain() -> Runnable:

    chat_model  = get_model_chain(
        structured_output=Notification,
    )

    summarization_chain: Runnable = {
        "content": itemgetter("message"),
    } | summary_messages_template | chat_model

    return summarization_chain.with_types(
        input_type=NotificationDTO,
    )