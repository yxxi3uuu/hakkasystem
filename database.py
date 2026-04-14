from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# 資料庫連線字串 (格式: postgresql+asyncpg://用戶名:密碼@主機:埠號/資料庫名)
DATABASE_URL = "postgresql+asyncpg://user:hakka2026@localhost/dbname"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Dependency: 用於 FastAPI 的注入，確保每次請求結束後會關閉連線
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
