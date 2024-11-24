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
from supabase import create_client, Client
from langchain_core.output_parsers import StrOutputParser
from app.models.dismissal_evaluator import NotificationDismissal
from app.prompts.reduce import reduce_messages_template
from app.prompts.dismissal_evaluator import evaluator_messages_template


from app.chat_models.default import get_model_chain
from app.dtos.notification import NotificationDTO
from supabase import create_client, Client

import requests

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


def dismiss_push(push: dict):
    KEY = os.getenv("PUSHBULLET_API_KEY")

    headers = {
        "Access-Token": KEY,
        "Content-Type": "application/json",
    }

    json_data = {
        "push": {
            "notification_id": push["notification_id"],
            "notification_tag": push["notification_tag"],
            "package_name": push["package_name"],
            "source_user_iden": push["source_user_iden"],
            "type": "dismissal",
        },
        "type": "push",
    }
    response = requests.post(
        "https://api.pushbullet.com/v2/ephemerals", headers=headers, json=json_data
    )
    print("Notification dismiss response status code", response.status_code)


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
    chat_model = get_model_chain()
    evaluator_model = get_model_chain(
        structured_output=NotificationDismissal,
        temperature=0,
    )

    connection_args = {
        "uri": os.environ.get("ZILLIZ_CLOUD_URI"),
        "user": os.environ.get("ZILLIZ_CLOUD_USER"),
        "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        "secure": True,
    }

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name="notifications",
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    categories_vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name="categories",
    )

    categories_retriever = categories_vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    def check_if_input_is_dismisable(input: dict):
        other_categories = [
            doc.metadata.get("category") for doc in input["similar_notifications"]
        ]

        supabase: Client = create_client(supabase_url, supabase_key)

        active_categories = (
            supabase.table("categoriesConfig")
            .select("name", "description")
            .eq('active', True)
            .execute()
        ).data

        categories_names = set([category["name"] for category in active_categories])

        other_categories_freq = {
            category: other_categories.count(category) for category in set(other_categories)

        }

        expected_category = max(other_categories_freq, key=lambda x: other_categories_freq[x])
        
        input["expected_category"] = expected_category
        input["similar_categories"] = list(categories_names)
        input["is_dismissible"] = not(bool(
            expected_category in categories_names
        )) and len(categories_names) > 0 and expected_category != ""
        return input
    
    def _check_active_inferred_categories(input: dict):
        supabase: Client = create_client(supabase_url, supabase_key)


        inferred_categories_docs = input["inferred_categories"]

        observed_categories = []

        for idx, doc in enumerate(inferred_categories_docs):
            category_name = doc.metadata.get("category")
            active_category = (
                supabase.table("categoriesConfig")
                .select("name", "description")
                .eq('active', True)
                .eq('name', category_name)
                .execute()
            ).data

            if active_category:
                observed_categories.append({
                    "name": category_name,
                    "description": active_category[idx].metdata["description"]
                })
        
        input["inferred_categories"] = observed_categories
        input["final_message"] = input["original_input"]["final_message"]
        input["is_dismissible"] = input["original_input"]["is_dismissible"]

        return input
    
    def _get_fixed_active_categories(input: list):
        supabase: Client = create_client(supabase_url, supabase_key)

        active_categories = (
            supabase.table("categoriesConfig")
            .select("name", "description")
            .eq('active', True)
            .eq("type", "fixed")
            .execute()
        ).data

        return input + active_categories
    
    def _manage_notification(input: dict):
        dismissal_resolution: NotificationDismissal = input["is_dismissible"]

        message_data = {
            "title": input["original_input"]["original_input"]["original_input"]["title"],
            "message": input["original_input"]["original_input"]["original_input"]["message"],
            "is_dismissable": dismissal_resolution.is_dismissible,
            "reason": dismissal_resolution.reason,
            "confidence": dismissal_resolution.confidence,
            "app": input["original_input"]["original_input"]["original_input"]["app_name"],
        }
        insert_message_to_supabase(message_data)

        if dismissal_resolution.is_dismissible:
            dismiss_push(
                push=input["original_input"]["original_input"]["original_input"]
            )
        return input

    classification_chain: Runnable = (
        RunnableLambda(func=partial(store_message, vector_store=vector_store))
        | RunnableParallel(
            {
                "original_input": RunnablePassthrough(),
                "similar_notifications": itemgetter("final_message") | retriever,
                "final_message": itemgetter("final_message")
            }
        )
        | RunnableLambda(check_if_input_is_dismisable)
        | {
            "original_input": RunnablePassthrough(),
            "inferred_categories": itemgetter("final_message") | RunnableLambda(
                lambda x: [x]
            ) | reduce_messages_template | chat_model | StrOutputParser() | categories_retriever,
        } | RunnableLambda(_check_active_inferred_categories)
        | {
            "original_input": RunnablePassthrough(),
            "final_message": itemgetter("final_message"),
            "is_dismissible": {
                "inferred_categories" : itemgetter("inferred_categories") | RunnableLambda(_get_fixed_active_categories),
                "final_message": itemgetter("final_message"),
                "is_dismissible": itemgetter("is_dismissible"),
                "schema": RunnableLambda(lambda _: NotificationDismissal.model_json_schema()),
            } | evaluator_messages_template | evaluator_model ,
        } | RunnableLambda(
            _manage_notification
            )
    ).with_types(
        input_type=NotificationDTO,
    )

    return classification_chain
