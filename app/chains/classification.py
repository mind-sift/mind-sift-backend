import os
from app.dtos.notification import NotificationDTO
from langchain_core.runnables import Runnable
from langchain_milvus.vectorstores import Milvus
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from functools import partial
from langchain_core.runnables import RunnableLambda

def store_message(input: dict, vector_store: VectorStore) -> dict:

    copy_input = input.copy()
    message = copy_input.pop("message")
    pk = copy_input.pop("id")

    notification_document = Document(
        id=pk,
        page_content=message,
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