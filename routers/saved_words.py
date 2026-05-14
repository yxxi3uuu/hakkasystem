import os
import uuid
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db
from models import SavedWord, Activity
import datetime

router = APIRouter(prefix="/api/saved_words", tags=["saved_words"])

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("")
async def save_word(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    label_zh: str = Form(...),
    label_hakka: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Save file to static/uploads
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        filename = f"{uuid.uuid4().hex}.{file_ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        image_path = f"/static/uploads/{filename}"
        
        # Save to database
        new_word = SavedWord(
            user_id=user_id,
            image_path=image_path,
            label_zh=label_zh,
            label_hakka=label_hakka
        )
        db.add(new_word)
        
        # Log activity
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        activity = Activity(
            user_id=user_id,
            icon="📷",
            title=f"拍照學習: {label_zh}",
            score=10,
            created_at=today
        )
        db.add(activity)
        
        await db.commit()
        await db.refresh(new_word)
        
        return {"status": "success", "id": new_word.id, "image_path": image_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
async def get_saved_words(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    # Get all saved words for current user
    result = await db.execute(
        select(SavedWord).where(SavedWord.user_id == user_id)
    )
    words = result.scalars().all()
    return [{"id": w.id, "image_path": w.image_path, "label_zh": w.label_zh, "label_hakka": w.label_hakka} for w in words]
