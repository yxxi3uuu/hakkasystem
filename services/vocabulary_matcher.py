"""Conservative matching from recognized Chinese names to official words."""

from __future__ import annotations

import re

from sqlalchemy import case, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import CertificationWord, WordAlias


def normalize_chinese_name(value: str) -> str:
    value = re.sub(r"\s+", "", value or "")
    return value.strip("，。！？、；：,.!?;:")


async def find_word_candidates(
    db: AsyncSession,
    name_zh: str,
    dialect: str = "四縣腔",
    limit: int = 5,
) -> list[CertificationWord]:
    normalized = normalize_chinese_name(name_zh)
    if not normalized:
        return []

    alias_word_ids = select(WordAlias.word_id).where(
        WordAlias.alias_text == normalized,
    )
    exact_rank = case(
        (CertificationWord.zh_meaning == normalized, 0),
        (CertificationWord.hakka_word == normalized, 1),
        (CertificationWord.word_id.in_(alias_word_ids), 2),
        else_=3,
    )
    statement = (
        select(CertificationWord)
        .where(
            CertificationWord.dialect == dialect,
            or_(
                CertificationWord.zh_meaning == normalized,
                CertificationWord.hakka_word == normalized,
                CertificationWord.word_id.in_(alias_word_ids),
            ),
        )
        .order_by(exact_rank, CertificationWord.word_id)
        .limit(max(1, min(limit, 20)))
    )
    return list((await db.execute(statement)).scalars().all())


async def find_best_word(
    db: AsyncSession,
    name_zh: str,
    aliases: list[str] | None = None,
    dialect: str = "四縣腔",
) -> CertificationWord | None:
    for candidate_name in [name_zh, *(aliases or [])]:
        matches = await find_word_candidates(
            db,
            candidate_name,
            dialect=dialect,
            limit=1,
        )
        if matches:
            return matches[0]
    return None


async def find_category_recommendations(
    db: AsyncSession,
    anchor: CertificationWord | None,
    limit: int = 5,
) -> list[CertificationWord]:
    """Return verified words from the anchor's official vocabulary category."""
    if not anchor or not anchor.category:
        return []

    level_rank = case(
        (CertificationWord.certification_level == "基礎級", 0),
        (CertificationWord.certification_level == "中級", 1),
        (CertificationWord.certification_level == "高級", 2),
        else_=3,
    )
    statement = (
        select(CertificationWord)
        .where(
            CertificationWord.dialect == anchor.dialect,
            CertificationWord.category == anchor.category,
            CertificationWord.word_id != anchor.word_id,
            CertificationWord.is_verified.is_(True),
        )
        .order_by(level_rank, CertificationWord.word_id)
        .limit(max(1, min(limit, 5)))
    )
    return list((await db.execute(statement)).scalars().all())
