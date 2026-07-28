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


def _created_at_value(word: SavedWord) -> str:
    created_at = getattr(word, "created_at", None)
    return created_at.isoformat() if created_at else ""


def _word_occurrence(word: SavedWord) -> dict:
    return {
        "id": word.id,
        "image_path": word.image_path,
        "created_at": _created_at_value(word),
        "source": getattr(word, "source", "yolo") or "yolo",
    }


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

    source: str = Form("yolo"),   # yolo | ocr
    word_id: int | None = Form(None),
    recognition_id: int | None = Form(None),
    dialect: str = Form(""),

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
            sentence_audio_path=sentence_audio_path,

            source=source,
            word_id=word_id,
            recognition_id=recognition_id,
            dialect=dialect,
        )

        db.add(new_word)

        icon = "🔤" if source == "ocr" else "📷"
        title_prefix = "文字辨識" if source == "ocr" else "拍照學習"
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        activity = Activity(
            user_id=user_id,
            icon=icon,
            title=f"{title_prefix}: {label_zh[:10]}",
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
            "sentence_audio_path": sentence_audio_path,
            "source": source,
            "word_id": word_id,
            "recognition_id": recognition_id,
            "dialect": dialect,
            "created_at": _created_at_value(new_word),
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
        .order_by(SavedWord.created_at.desc().nullslast(), SavedWord.id.desc())
    )

    words = result.scalars().all()

    # 計算每個 label_zh 出現幾次
    count_map: dict[str, int] = {}
    occurrence_map: dict[str, list[dict]] = {}
    for w in words:
        count_map[w.label_zh] = count_map.get(w.label_zh, 0) + 1
        occurrence_map.setdefault(w.label_zh, []).append(_word_occurrence(w))

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
                "sentence_audio_path": getattr(w, "sentence_audio_path", ""),

                "count": count_map.get(w.label_zh, 1),
                "occurrences": occurrence_map.get(w.label_zh, []),
                "source": getattr(w, "source", "yolo"),
                "word_id": getattr(w, "word_id", None),
                "recognition_id": getattr(w, "recognition_id", None),
                "dialect": getattr(w, "dialect", ""),
                "created_at": _created_at_value(w),
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
            "sentence_audio_path": getattr(w, "sentence_audio_path", ""),
            "source": getattr(w, "source", "yolo") or "yolo",
            "word_id": getattr(w, "word_id", None),
            "recognition_id": getattr(w, "recognition_id", None),
            "dialect": getattr(w, "dialect", ""),
            "created_at": _created_at_value(w),
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
