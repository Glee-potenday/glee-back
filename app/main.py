from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.auth.auth_router import router as auth_router
from app.suggester.suggester_router import router as analyze_router
from app.history.history_router import router as history_router
from app.core.settings import settings
from app.utils import mongo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await mongo.set_indexes()
    yield


# ✅ Lifespan을 FastAPI에 연결하여 사용
app = FastAPI(lifespan=lifespan)

# ✅ FastAPI 인스턴스 생성 후 라우터 추가
app.include_router(auth_router)
app.include_router(analyze_router)
app.include_router(history_router)

# ✅ CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ SessionMiddleware 추가
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,  # 반드시 변경할 것!
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
