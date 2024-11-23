import os
from app.dtos.notification import NotificationDTO
from langchain_core.runnables import Runnable
from langchain_milvus.vectorstores import Milvus
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from functools import partial
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

notification_prompt = """
<NOTIFICATION>

    <Title>
        {title}
    </Title>

    <Message>
        {message}
    </Message>

</NOTIFICATION>
"""

notification_templates = PromptTemplate.from_template(notification_prompt)

def store_message(input: dict, vector_store: VectorStore) -> dict:

    copy_input = input.copy()
    copy_input["timestamp"] = float(copy_input["timestamp"])
    final_message = notification_templates.format(**copy_input)
    pk = copy_input.pop("id")

    notification_document = Document(
        id=pk,
        page_content=final_message,
        metadata={
            **copy_input,
        },
    )

    vector_store.add_documents(
        documents=[notification_document],
        ids=[pk]
    )

    return input

def get_classification_chain() -> Runnable:

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1"
    )

    connection_args = {
        "uri": os.environ.get("ZILLIZ_CLOUD_URI"),
        "user": os.environ.get("ZILLIZ_CLOUD_USER"),
        "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        "secure": True,
    }

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name="notifications",
    )
    

    classification_chain: Runnable =  RunnableLambda(
        func=partial(store_message, vector_store=vector_store)
    ).with_types(
        input_type=NotificationDTO,
    )

    return classification_chain