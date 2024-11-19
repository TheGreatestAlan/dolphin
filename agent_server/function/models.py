from pydantic import BaseModel
from typing import List

class CreateItemsParams(BaseModel):
    container: str
    items: List[str]

class FindLocationParams(BaseModel):
    item_name: str

# Add other parameter models as needed
