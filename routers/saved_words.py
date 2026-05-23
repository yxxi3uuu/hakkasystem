import os
import uuid
import shutil
import datetime

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from database import get_db
from models import SavedWord, Activity

router = APIRouter(prefix="/api/saved_words", tags=["saved_words"])

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("")
async def save_word(
    user_id: int = Form(...),
    file: UploadFile = File(...),

    label_zh: str = Form(...),
    label_hakka: str = Form(...),
    label_pinyin: str = Form(""),
    audio_path: str = Form(""),

    labels_json: str = Form(""),

    sentence_zh: str = Form(""),
    sentence_hakka: str = Form(""),
    sentence_audio_path: str = Form(""),

    db: AsyncSession = Depends(get_db)
):
    try:
        file_ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
        filename = f"{uuid.uuid4().hex}.{file_ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image_path = f"/static/uploads/{filename}"

        new_word = SavedWord(
            user_id=user_id,
            image_path=image_path,

            label_zh=label_zh,
            label_hakka=label_hakka,
            label_pinyin=label_pinyin,
            audio_path=audio_path,

            labels_json=labels_json,

            sentence_zh=sentence_zh,
            sentence_hakka=sentence_hakka,
            sentence_audio_path=sentence_audio_path
        )

        db.add(new_word)

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

        return {
            "status": "success",
            "id": new_word.id,
            "image_path": image_path,
            "audio_path": audio_path,
            "sentence_audio_path": sentence_audio_path
        }

    except Exception as e:
        await db.rollback()
        print(f"❌ 儲存單字發生異常報錯: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unique")
async def get_unique_saved_words(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(SavedWord)
        .where(SavedWord.user_id == user_id)
        .order_by(SavedWord.id.asc())
    )

    words = result.scalars().all()

    seen = set()
    unique_words = []

    for w in words:
        if w.label_zh not in seen:
            seen.add(w.label_zh)

            unique_words.append({
                "id": w.id,
                "image_path": w.image_path,

                "label_zh": w.label_zh,
                "label_hakka": w.label_hakka,
                "label_pinyin": getattr(w, "label_pinyin", ""),
                "audio_path": getattr(w, "audio_path", ""),

                "labels_json": getattr(w, "labels_json", ""),

                "sentence_zh": w.sentence_zh,
                "sentence_hakka": w.sentence_hakka,
                "sentence_audio_path": getattr(w, "sentence_audio_path", "")
            })

    return unique_words


@router.get("")
async def get_saved_words(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(SavedWord).where(SavedWord.user_id == user_id)
    )

    words = result.scalars().all()

    return [
        {
            "id": w.id,
            "image_path": w.image_path,

            "label_zh": w.label_zh,
            "label_hakka": w.label_hakka,
            "label_pinyin": getattr(w, "label_pinyin", ""),
            "audio_path": getattr(w, "audio_path", ""),

            "labels_json": getattr(w, "labels_json", ""),

            "sentence_zh": w.sentence_zh,
            "sentence_hakka": w.sentence_hakka,
            "sentence_audio_path": getattr(w, "sentence_audio_path", "")
        }
        for w in words
    ]


@router.delete("/{word_id}")
async def delete_saved_word(
    word_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(SavedWord).where(
            SavedWord.id == word_id,
            SavedWord.user_id == user_id
        )
    )

    word = result.scalar_one_or_none()

    if not word:
        raise HTTPException(status_code=404, detail="單字不存在或無權限刪除")

    if word.image_path:
        file_path = word.image_path.lstrip("/")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

    # 如果也想刪掉單字語音，可以保留這段
    if getattr(word, "audio_path", ""):
        audio_file = word.audio_path.lstrip("/")
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception:
                pass

    await db.delete(word)
    await db.commit()

    return {
        "status": "success",
        "message": f"已刪除單字：{word.label_zh}"
    }