from fastapi import APIRouter, HTTPException, Request
import os
from db.supabase import Supabase


Supabase_URL = os.getenv("SUPABASE_PROJECT_URL")
Supabase_KEY = os.getenv("SUPABASE_API_KEY")

router = APIRouter(
    prefix="/supabase",
    tags=["Supabase"],
)

@router.get("/get-bucket")
async def get_database_url(request: Request):
    try:
        supabase = request.app.state.supabase_client
        response = supabase.storage.from_("Database").download("Quran.zip")  # Assume this returns bytes
        if isinstance(response, bytes):
            with open("Quran.zip", "wb") as file:
                file.write(response)
            return {"status": "success", "message": "File downloaded successfully"}
        else:
            return {"status": "error", "message": "Failed to download the file"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

