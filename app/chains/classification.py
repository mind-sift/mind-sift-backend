import os
from functools import partial
from operator import itemgetter

from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStore
from langchain_milvus.vectorstores import Milvus

from app.dtos.notification import NotificationDTO
from supabase import create_client, Client

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

    vector_store.add_documents(documents=[notification_document], ids=[pk])

    input["final_message"] = final_message

    return input


def insert_message_to_supabase(message_data: dict):
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
    supabase.table("messages").upsert(json=message_data).execute()


def get_classification_chain() -> Runnable:

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

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

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    def check_if_input_is_dismisable(input: dict):
        original_input_category = input["original_input"].get("category")
        other_categories = [
            doc.metadata.get("category") for doc in input["similar_nofications"]
        ]
        input["is_dismissable"] = (
            original_input_category is not None
            and original_input_category in other_categories
        )

        message_data = {
            "title": input["original_input"]["title"],
            "message": input["original_input"]["message"],
            "is_dismissable": input["is_dismissable"],
        }
        insert_message_to_supabase(message_data)

        return input

    classification_chain: Runnable = (
        RunnableLambda(func=partial(store_message, vector_store=vector_store))
        | RunnableParallel(
            {
                "original_input": RunnablePassthrough(),
                "similar_nofications": itemgetter("final_message") | retriever,
            }
        )
        | RunnableLambda(check_if_input_is_dismisable)
    ).with_types(
        input_type=NotificationDTO,
    )

    return classification_chain
