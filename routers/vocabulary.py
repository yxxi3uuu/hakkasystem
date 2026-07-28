"""Read-only APIs for official certification vocabulary."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import CertificationWord
from services.vocabulary_matcher import find_word_candidates


router = APIRouter(prefix="/api/words", tags=["certification-vocabulary"])


class CertificationWordResponse(BaseModel):
    word_id: int
    source_word_code: str
    hakka_word: str
    zh_meaning: str
    dialect: str
    pinyin: str
    certification_level: str
    category: str
    part_of_speech: list[str]
    example_sentence: str
    example_translation: str
    audio_url: str
    source: str
    source_version: str
    is_verified: bool


class WordSearchResponse(BaseModel):
    query: str
    total: int
    items: list[CertificationWordResponse]


def serialize_word(word: CertificationWord) -> CertificationWordResponse:
    part_of_speech = [
        value
        for value in (
            word.part_of_speech_primary,
            word.part_of_speech_secondary,
        )
        if value
    ]
    return CertificationWordResponse(
        word_id=word.word_id,
        source_word_code=word.source_word_code,
        hakka_word=word.hakka_word,
        zh_meaning=word.zh_meaning,
        dialect=word.dialect,
        pinyin=word.pinyin,
        certification_level=word.certification_level,
        category=word.category,
        part_of_speech=part_of_speech,
        example_sentence=word.example_sentence,
        example_translation=word.example_translation,
        audio_url=word.audio_url,
        source=word.source,
        source_version=word.source_version,
        is_verified=word.is_verified,
    )


@router.get("/search", response_model=WordSearchResponse)
async def search_words(
    q: str = Query(..., min_length=1, max_length=80),
    dialect: str = Query("四縣腔", min_length=1, max_length=20),
    level: str | None = Query(None, max_length=20),
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    query = q.strip()
    filters = [
        CertificationWord.dialect == dialect,
        or_(
            CertificationWord.zh_meaning.contains(query),
            CertificationWord.hakka_word.contains(query),
            CertificationWord.pinyin.ilike(f"%{query}%"),
        ),
    ]
    if level:
        filters.append(CertificationWord.certification_level == level)

    total = (
        await db.execute(
            select(func.count(CertificationWord.word_id)).where(*filters)
        )
    ).scalar_one()
    words = (
        await db.execute(
            select(CertificationWord)
            .where(*filters)
            .order_by(CertificationWord.word_id)
            .limit(limit)
        )
    ).scalars().all()
    return WordSearchResponse(
        query=query,
        total=total,
        items=[serialize_word(word) for word in words],
    )


@router.get("/match", response_model=list[CertificationWordResponse])
async def match_word(
    name_zh: str = Query(..., min_length=1, max_length=80),
    dialect: str = Query("四縣腔", min_length=1, max_length=20),
    limit: int = Query(5, ge=1, le=5),
    db: AsyncSession = Depends(get_db),
):
    words = await find_word_candidates(
        db,
        name_zh,
        dialect=dialect,
        limit=limit,
    )
    return [serialize_word(word) for word in words]


@router.get("/{word_id}", response_model=CertificationWordResponse)
async def get_word(
    word_id: int,
    db: AsyncSession = Depends(get_db),
):
    word = await db.get(CertificationWord, word_id)
    if not word:
        raise HTTPException(status_code=404, detail="詞庫中找不到此詞彙")
    return serialize_word(word)
