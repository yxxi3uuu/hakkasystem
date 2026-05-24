"""檢查 practice 相關的資料庫狀態"""
import asyncio
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from models import SavedWord, User
import os

engine = create_async_engine(os.getenv("DATABASE_URL"))
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def main():
    async with AsyncSessionLocal() as db:
        # 查所有使用者
        users = (await db.execute(select(User))).scalars().all()
        print(f"使用者數量: {len(users)}")
        for u in users:
            print(f"  user_id={u.id}, name={u.name}, email={u.email}")

        # 查所有儲存的單字
        words = (await db.execute(select(SavedWord))).scalars().all()
        print(f"\n儲存的單字數量: {len(words)}")
        for w in words:
            has_audio = bool(w.audio_path)
            has_image = bool(w.image_path)
            audio_exists = False
            if w.audio_path:
                audio_exists = Path(w.audio_path.lstrip("/")).exists()
            print(f"  id={w.id} user={w.user_id} zh={w.label_zh} "
                  f"audio={'✅' if has_audio else '❌'} "
                  f"audio_file={'✅' if audio_exists else '❌(檔案不存在)'} "
                  f"image={'✅' if has_image else '❌'}")

asyncio.run(main())
