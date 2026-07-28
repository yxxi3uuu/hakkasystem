"""Dry-run or import official certification vocabulary ODS files."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import AsyncSessionLocal, Base, engine
from models import CertificationWord, VocabularySource
from services.vocabulary_importer import (
    ParsedVocabulary,
    parse_vocabulary_ods,
    parsed_report_json,
    word_as_database_dict,
)

# Importing thousands of Unicode-rich rows with SQL echo enabled produces very
# large logs and can fail to render rare Hakka characters on legacy consoles.
engine.echo = False


def write_report(parsed: ParsedVocabulary, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{parsed.path.stem}.import-report.json"
    report_path.write_text(parsed_report_json(parsed), encoding="utf-8")
    return report_path


async def import_parsed(parsed_items: list[ParsedVocabulary]) -> dict[str, int]:
    imported_sources = 0
    imported_words = 0
    skipped_sources = 0

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        for parsed in parsed_items:
            existing = (
                await session.execute(
                    select(VocabularySource).where(
                        VocabularySource.checksum == parsed.checksum
                    )
                )
            ).scalar_one_or_none()
            if existing:
                skipped_sources += 1
                continue

            source = VocabularySource(
                source_name=parsed.source_name,
                source_version=parsed.source_version,
                original_filename=parsed.path.name,
                checksum=parsed.checksum,
                import_report_json=json.dumps(
                    parsed.report,
                    ensure_ascii=False,
                ),
            )
            session.add(source)
            await session.flush()

            session.add_all(
                [
                    CertificationWord(
                        source_id=source.id,
                        **word_as_database_dict(word),
                    )
                    for word in parsed.words
                ]
            )
            await session.commit()
            imported_sources += 1
            imported_words += len(parsed.words)

    return {
        "imported_sources": imported_sources,
        "imported_words": imported_words,
        "skipped_sources": skipped_sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "vocabulary" / "reports",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write to the configured database. Without this flag, dry-run only.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    parsed_items: list[ParsedVocabulary] = []

    for path in args.files:
        try:
            parsed = parse_vocabulary_ods(path)
        except ValueError as error:
            print(f"SKIP {path.name}: {error}")
            continue
        parsed_items.append(parsed)
        report_path = write_report(parsed, args.report_dir)
        print(
            f"{parsed.source_version}: {len(parsed.words)} importable, "
            f"{len(parsed.flagged_rows)} flagged, report={report_path}"
        )

    if not parsed_items:
        raise SystemExit("No certification vocabulary sheets were found.")

    if not args.apply:
        print("Dry-run complete. Database was not changed.")
        return

    result = await import_parsed(parsed_items)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
