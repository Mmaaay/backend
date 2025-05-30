from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as exc:
            if exc.status_code == 401:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized - Please login to access this resource"}
                )
            raise exc
