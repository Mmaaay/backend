import os

from constants import SUPABASE_KEY, SUPABASE_URL
from fastapi import HTTPException
from supabase import Client, create_client


class Supabase:
    def __init__(self, url: str, key: str):
        if not url or not key:
            raise ValueError("Supabase URL and key must be provided.")
        try:
            self.client: Client = create_client(url, key)
            print("Connected to Supabase successfully.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Supabase: {e}")

    def get_client(self) -> Client:
        return self.client


    def close_connection(self):
        self.client.close()
        print("Supabase connection closed.")
