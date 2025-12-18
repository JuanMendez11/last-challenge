from pydantic import BaseModel

# Input
class Request(BaseModel):
    content: str

# Output
class Response(BaseModel):
    content: str