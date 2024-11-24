import uuid
from hashlib import md5

def category_id(category: str) -> str:
    hash_md5 = md5(category.encode()).hexdigest()

    # Convert MD5 hash to UUID
    uuid_value = str(uuid.UUID(hash_md5))
    return uuid_value