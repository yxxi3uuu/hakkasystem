from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db, engine, Base
from models import User
from contextlib import asynccontextmanager
import os
DATABASE_URL = "postgresql+asyncpg://user:hakka2026@localhost:5432/dbname"

app = FastAPI()

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("App started")

    yield  # 👉 中間是 API 運行期間

    # shutdown
    print("App stopped")

app = FastAPI(lifespan=lifespan)

@app.get("/users")
async def read_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return users

@app.post("/users")
async def create_user(name: str, email: str, db: AsyncSession = Depends(get_db)):
    new_user = User(name=name, email=email)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

@app.get("/test-db")
def test_db():
    return {"status": "ok"}
