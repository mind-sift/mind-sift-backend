from langchain_aws.chat_models import ChatBedrock
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from typing import Type

def get_model_chain(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    structured_output: Type[BaseModel] | None = None,
    *args,
    **kwargs,
) -> Runnable:
    model = ChatBedrock(
        model=model,
        region="us-west-2",
        *args,
        **kwargs,
    )
    runnable = model

    if structured_output is not None:
        runnable = model.with_structured_output(
            schema=structured_output,
        )
    
    return runnable


