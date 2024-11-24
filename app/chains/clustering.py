import json
from operator import itemgetter
import os
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

import time
from pymilvus import connections, Collection
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
from langchain_core.documents import Document
from app.dtos.clustering import ClusteringDTO
from app.prompts.reduce import reduce_messages_template
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus.vectorstores import Milvus
from langchain_aws.embeddings import BedrockEmbeddings
from supabase import create_client, Client


from app.chat_models.default import get_model_chain
from app.prompts.rule_description import rule_description_messages_template
from app.utils import category_id

SEVEN_DAYS = 7 * 24 * 60 * 60

def get_vectors(
        collection_name: str = "notifications",
        time_window: int = SEVEN_DAYS
        ) -> pd.DataFrame:
    """
    Retrieve vectors and primary keys (PK) from the past week stored in a Milvus collection on Zilliz Cloud.

    Args:
        collection_name (str): The name of the Milvus collection to query.

    Returns:
        pd.DataFrame: A DataFrame containing the vectors and PKs.
    """
    
    # Load the collection
    collection = get_vector_store_client(collection_name)
    
    # Current time and one week ago
    current_time = time.time()
    one_week_ago = current_time - time_window

    fields = [
        "pk",
        "vector",
        "text"
    ]

    
    # Query vectors and PKs from the past week
    results = collection.query(
        expr=f"timestamp >= {one_week_ago} and timestamp <= {current_time}",
        output_fields=fields,
        limit=30
    )
    
    # Parse results
    data = [
        {
            field: result[field]
            for field in fields
        }
        for result in results
    ]
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)
    return df

def get_vector_store_client(collection_name: str = "notifications") -> Collection:

    connection_args = {
        "uri": os.environ.get("ZILLIZ_CLOUD_URI"),
        "user": os.environ.get("ZILLIZ_CLOUD_USER"),
        "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        "secure": True,
    }

    connections.connect(
        **connection_args,
    )
    
    collection = Collection(collection_name)

    return collection

def upsert_vectors(
    pk: str,
    data: dict,
    fields: list | None = None
    ):
    """
    Upsert a document into the Milvus collection.

    Args:
        pk (str): The primary key of the document.
        data (dict): The data to upsert.
    """
    if fields is None:
        fields = [
            "pk",
            "vector",
            "notification_id",
            "notification_tag",
            "package_name",
            "source_user_iden",
            "title",
            "message",
            "app_name",
            "timestamp",
            "category",
            "text"
        ]

    # Connect to the Milvus collection
    collection = get_vector_store_client()

    output = collection.query(
        expr=f"pk == '{pk}'",
        output_fields=fields
    )
    
    # Upsert the document
    collection.upsert(
        data={
            "pk": pk,
            **output[0],
            **data
        },
    )
    

def perform_dbscan_cosine_clustering(df, eps=0.1, min_samples=5):
    """
    Perform DBSCAN clustering using cosine similarity.

    Args:
        df (pd.DataFrame): A DataFrame containing `vector` and `pk` columns.
        eps (float): The maximum cosine distance between two points to be considered neighbors.
        min_samples (int): The minimum number of points required to form a cluster.

    Returns:
        pd.DataFrame: The original DataFrame with an added `cluster` column.
    """
    # Convert vectors to a NumPy array
    vectors = np.array(df["vector"].tolist())
    
    # Calculate cosine distances
    distance_matrix = cosine_distances(vectors)

    # Run DBSCAN clustering with precomputed distances
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(distance_matrix)
    
    # Add cluster labels to the DataFrame
    df["cluster"] = cluster_labels
    
    # Print the number of clusters
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Exclude noise (-1)
    print(f"Number of clusters found: {n_clusters}")
    return df

def clusters_of_documents() -> dict[int, list[Document]]:

    vectors_df = get_vectors(
        collection_name="notifications",
        time_window=30
    )

    clustered_df = perform_dbscan_cosine_clustering(
        df=vectors_df,
        eps=0.2,
        min_samples=5
    )

    source_data = {
        cluster: clustered_df.loc[clustered_df["cluster"] == cluster, ~clustered_df.columns.isin(["vector"])].to_dict(orient="records")
        for cluster in clustered_df["cluster"].unique()
    }

    final_document_groups: dict[int, list[Document]] = {
        int(cluster): [
            Document(
                id=str(doc["pk"]),
                page_content=doc["text"],
                metadata={
                    key: value
                    for key, value in doc.items() if key != "pk" or key != "text"
                }
            )
            for doc in docs
        ]
        for cluster, docs in source_data.items()
    }

    return final_document_groups


def get_clusters_chain() -> Runnable:

    chat_model = get_model_chain(
        temperature=0.3,
        #top_p=0.95,
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1"
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
        collection_name="categories",
    )

    max_similar = 9

    categories_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": max_similar,
            "score_threshold": 0.8
        }
    )

    chain: Runnable = {
            "documents": itemgetter("documents")
        } | {
            "category": reduce_messages_template | chat_model | StrOutputParser(),
            "description": rule_description_messages_template | chat_model | StrOutputParser()
        } | RunnablePassthrough()

    def _get_clusters(_, **kwargs):

        input_data = clusters_of_documents()
        input_data = {
            key: value
            for key, value in input_data.items()
            if key != -1
        }

        values = zip(chain.batch(
            inputs=[
                {
                    "documents": cluster
                }
                for cluster
                in input_data.values()
                ],
            **kwargs
        ), input_data.values())



        return [
            {
                "category": category["category"],
                "description": category["description"],
                "documents": original_docs
            }
            for category, original_docs in values
        ]
    
    def _update_metadata(inputs: list):

        categories = []
        
        for category_elements in inputs:
            sources_pks = []
            for doc in category_elements["documents"]:
                sources_pks.append(doc.id)
                upsert_metadata = {
                    **doc.metadata,
                    "category": category_elements["category"],
                }
                upsert_metadata.pop("cluster")
                upsert_vectors(
                    pk=doc.id,
                    data=upsert_metadata
                )

            category_descriptor: Document = Document(
                id=category_id(category_elements["category"]),
                page_content=category_elements["category"],
                metadata={
                    "generated_by": json.dumps(sources_pks),
                }
            )
        
            categories.append(category_descriptor)
        
        return categories
    
    def _store_categories(inputs: list):
        """
        Store the categories in the Milvus collection.
        """
        if inputs:
            vector_store.add_documents(
                documents=inputs,
                ids=[category.id for category in inputs]
            )
        return inputs
    
    def _reduce_candidates(inputs: dict):
            """
            Reduce the candidate if max_similar is reached.
            Discarding the category, due to the high similarity. 
            """

            output = {
                "route": "store" if len(inputs["candidates"]) < max_similar else "discard"
            }
            return output


    def _route_categories(inputs: list):
        

        categories_parallelizable = {
                "category": itemgetter("category"),
                "candidates": (itemgetter("category") | categories_retriever).with_fallbacks(
                    fallbacks=[RunnableLambda(lambda x: [])]
                )
            }

        routing_chain: Runnable = categories_parallelizable | RunnableLambda(_reduce_candidates)


        routes = routing_chain.batch(inputs=[
            {
                "category": category["category"],
            }
            for category in inputs
        ])

        return routes
    
    def _allow_insertion(inputs: dict):
        """
        Allow insertion of the category if the route is "store".
        """

        return [
            category
            for category, route in zip(inputs["clusters"], inputs["routes"])
            if route["route"] == "store"
        ]
    
    def _update_user_options(inputs: list):
        """
        Update the user options in the Milvus collection.
        """
        supabase: Client = create_client(supabase_url, supabase_key)
        if inputs:
            payload = [
                    {
                        "id": category.id,
                        "name": category.page_content,
                        "description": inputs["original_categories"][idx]["description"],
                    }
                    for idx, category in enumerate(inputs["vectorized_categories"])
                ]

            response = supabase.table("categoriesConfig").upsert(
                json=payload
            ).execute()

            return response
        return inputs


    return (
            RunnableLambda(_get_clusters) |
            {
                "clusters": RunnablePassthrough(),
                "routes": RunnableLambda(_route_categories)
            } |
            RunnableLambda(_allow_insertion) |
            {
                "vectorized_categories": RunnableLambda(_update_metadata) | RunnableLambda(_store_categories),
                "original_categories": RunnablePassthrough()
            } | RunnableLambda(_update_user_options)
        ).with_types(
        input_type=ClusteringDTO,
    ) 
    



