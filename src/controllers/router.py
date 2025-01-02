from fastapi import APIRouter

# Create the router instance
chat_router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

# Export the router instance
__all__ = ['chat_router']
