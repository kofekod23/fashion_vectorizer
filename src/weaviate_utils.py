import weaviate
from weaviate.auth import AuthApiKey
from weaviate.collections.classes.config import DataType, Vectorizers

def connect_weaviate(cluster_url, api_key):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=AuthApiKey(api_key)
    )
    assert client.is_ready(), "Weaviate client is not ready!"
    return client

def create_collections_if_needed(client):
    if "Tenues_v2025_c" not in client.collections.list_all():
        client.collections.create(
            name="Tenues_v2025_c",
            properties=[
                {"name": "tenueId", "data_type": DataType.TEXT},
                {"name": "garmentCroppedIds", "data_type": DataType.TEXT_ARRAY},
                {"name": "garmentCroppedFiles", "data_type": DataType.TEXT_ARRAY},
                {"name": "catsAgg", "data_type": DataType.TEXT_ARRAY},
                {"name": "descAgg", "data_type": DataType.TEXT},
                {"name": "tenuePrimaryFile", "data_type": DataType.TEXT},
            ],
            vector_config=[
                {"name": "fclip", "vectorizer": {"vectorizer": Vectorizers.NONE}},
                {"name": "vit", "vectorizer": {"vectorizer": Vectorizers.NONE}},
            ]
        )
        print("Collection Tenues_v2025_c créée.")
    if "Vetements_v2025_c" not in client.collections.list_all():
        client.collections.create(
            name="Vetements_v2025_c",
            properties=[
                {"name": "origImageId", "data_type": DataType.TEXT},
                {"name": "image_original_file", "data_type": DataType.TEXT},
                {"name": "imageCroppedId", "data_type": DataType.TEXT},
                {"name": "categoryName", "data_type": DataType.TEXT},
                {"name": "imageCroppedFile", "data_type": DataType.TEXT},
                {"name": "score", "data_type": DataType.NUMBER},
            ],
            vector_config=[
                {"name": "fclip", "vectorizer": {"vectorizer": Vectorizers.NONE}},
                {"name": "vit", "vectorizer": {"vectorizer": Vectorizers.NONE}},
            ]
        )
        print("Collection Vetements_v2025_c créée.")

def insert_tenue(client, tenue_dict):
    client.collections.get("Tenues_v2025_c").data.insert(
        properties=tenue_dict["properties"],
        vector=tenue_dict["vectors"]
    )

def insert_vetement(client, vetement_dict):
    client.collections.get("Vetements_v2025_c").data.insert(
        properties=vetement_dict["properties"],
        vector=vetement_dict["vectors"]
    )
