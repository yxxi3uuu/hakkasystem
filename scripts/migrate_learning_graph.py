"""Create the certification vocabulary and learning graph schema.

This migration is intentionally idempotent. New installations get all tables
from SQLAlchemy metadata, while existing installations also receive the three
nullable compatibility columns added to saved_words.
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import Base, engine
import models  # noqa: F401 - registers every model with Base.metadata


SAVED_WORD_MIGRATIONS = (
    """
    ALTER TABLE saved_words
    ADD COLUMN IF NOT EXISTS word_id INTEGER
    REFERENCES certification_words(word_id) ON DELETE SET NULL
    """,
    """
    ALTER TABLE saved_words
    ADD COLUMN IF NOT EXISTS recognition_id INTEGER
    REFERENCES recognitions(recognition_id) ON DELETE SET NULL
    """,
    """
    ALTER TABLE saved_words
    ADD COLUMN IF NOT EXISTS dialect VARCHAR DEFAULT ''
    """,
    """
    ALTER TABLE saved_words
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ
    """,
    """
    ALTER TABLE saved_words
    ALTER COLUMN created_at SET DEFAULT NOW()
    """,
)


async def migrate() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        for statement in SAVED_WORD_MIGRATIONS:
            await connection.execute(text(statement))
    print("Certification vocabulary schema is ready.")


if __name__ == "__main__":
    asyncio.run(migrate())
