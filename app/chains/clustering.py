from operator import itemgetter
import os
from langchain_core.runnables import Runnable, RunnableLambda

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
from langchain.hub import pull

from app.chat_models.default import get_model_chain

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
    # Connect to Zilliz Cloud
    connections.connect(
        alias="default",
        uri=os.environ.get("ZILLIZ_CLOUD_URI"),
        user=os.environ.get("ZILLIZ_CLOUD_USER"),
        password=os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        secure=True
    )
    
    # Load the collection
    collection = Collection(collection_name)
    
    # Current time and one week ago
    current_time = time.time()
    one_week_ago = current_time - time_window
    
    # Query vectors and PKs from the past week
    results = collection.query(
        expr=f"timestamp >= {one_week_ago} and timestamp <= {current_time}",
        output_fields=["vector", "pk", "text"]
    )
    
    # Parse results
    data = [
        {
            "vector": result["vector"],
            "pk": result["pk"],
            "text": result["text"]
        }
        for result in results
    ]
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)
    return df


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
        time_window=SEVEN_DAYS
    )

    clustered_df = perform_dbscan_cosine_clustering(
        df=vectors_df,
        eps=0.1,
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
                metadata=doc
            )
            for doc in docs
        ]
        for cluster, docs in source_data.items()
    }

    return final_document_groups


def get_clusters_chain() -> Runnable:

    chat_model = get_model_chain()

    chain: Runnable = {
            "documents": itemgetter("documents")
        } | reduce_messages_template | chat_model | StrOutputParser()

    def _get_clusters(_, **kwargs):

        input_data = clusters_of_documents()

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
                "category": category,
                "documents": original_docs
            }
            for category, original_docs in values
        ]


    return RunnableLambda(_get_clusters).with_types(
        input_type=ClusteringDTO,
    ) 
    



