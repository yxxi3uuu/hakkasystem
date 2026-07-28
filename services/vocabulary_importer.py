"""Read and normalize official Hakka certification vocabulary ODS files."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
import json
import re
import xml.etree.ElementTree as ElementTree


TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
TABLE = f"{{{TABLE_NS}}}"
NAMESPACES = {"table": TABLE_NS}

LEVEL_BY_SHEET_PREFIX = {
    "基礎級": "基礎級",
    "中級": "中級",
    "高級": "高級",
}

REQUIRED_FIELDS = ("source_word_code", "hakka_word", "zh_meaning", "pinyin")


@dataclass
class NormalizedWord:
    source_word_code: str
    hakka_word: str
    zh_meaning: str
    dialect: str
    pinyin: str
    certification_level: str
    category_code: str
    category: str
    subcategory: str
    part_of_speech_primary: str
    part_of_speech_secondary: str
    example_sentence: str
    example_translation: str
    audio_url: str
    notes: str
    source: str
    source_version: str
    visualizable_type: str
    is_verified: bool
    normalization_status: str
    missing_fields: list[str]
    raw_data: dict[str, str]


@dataclass
class ParsedVocabulary:
    path: Path
    checksum: str
    source_name: str
    source_version: str
    sheet_name: str
    headers: list[str]
    words: list[NormalizedWord]
    flagged_rows: list[dict[str, Any]]
    report: dict[str, Any]


def file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_text(cell: ElementTree.Element) -> str:
    return "".join(cell.itertext()).replace("\xa0", " ").strip()


def read_ods_sheets(path: Path) -> dict[str, list[list[str]]]:
    try:
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("content.xml"))
    except (BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise ValueError(f"無法讀取 ODS 檔案：{path.name}") from error

    sheets: dict[str, list[list[str]]] = {}
    for table in root.findall(".//table:table", NAMESPACES):
        sheet_name = table.attrib.get(f"{TABLE}name", "")
        rows: list[list[str]] = []
        for row in table.findall("table:table-row", NAMESPACES):
            row_repeat = min(
                int(row.attrib.get(f"{TABLE}number-rows-repeated", "1")),
                1000,
            )
            values: list[str] = []
            for cell in list(row):
                if cell.tag not in (
                    f"{TABLE}table-cell",
                    f"{TABLE}covered-table-cell",
                ):
                    continue
                column_repeat = min(
                    int(cell.attrib.get(f"{TABLE}number-columns-repeated", "1")),
                    100,
                )
                values.extend([_cell_text(cell)] * column_repeat)

            while values and not values[-1]:
                values.pop()
            if any(values):
                rows.extend([values] * row_repeat)
        sheets[sheet_name] = rows
    return sheets


def detect_vocabulary_sheet(
    sheets: dict[str, list[list[str]]],
) -> tuple[str, str, list[list[str]]]:
    for sheet_name, rows in sheets.items():
        for prefix, level in LEVEL_BY_SHEET_PREFIX.items():
            if sheet_name.startswith(prefix) and rows:
                return sheet_name, level, rows
    raise ValueError("檔案中沒有找到基礎級、中級或高級詞彙工作表")


def _value(raw: dict[str, str], *headers: str) -> str:
    for header in headers:
        if header in raw:
            return raw[header].strip()
    return ""


def _split_category(value: str) -> tuple[str, str]:
    match = re.match(r"^\s*(\d+)\s*(.+?)\s*$", value)
    if not match:
        return "", value.strip()
    return match.group(1), match.group(2)


def _split_advanced_meaning(value: str) -> tuple[str, str]:
    for marker in ("例如：", "例如:", "例：", "例:"):
        if marker in value:
            meaning, example = value.split(marker, 1)
            return meaning.strip(), example.strip()
    return value.strip(), ""


def _source_version(path: Path, level: str) -> str:
    match = re.match(r"^(\d{3})\s*年度", path.name)
    year = match.group(1) if match else "unknown"
    return f"{year}-{level}-四縣腔"


def parse_vocabulary_ods(path: str | Path) -> ParsedVocabulary:
    source_path = Path(path).resolve()
    sheets = read_ods_sheets(source_path)
    sheet_name, level, rows = detect_vocabulary_sheet(sheets)
    headers = rows[0]
    source_version = _source_version(source_path, level)

    words: list[NormalizedWord] = []
    flagged_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows[1:], start=2):
        padded = row + [""] * max(0, len(headers) - len(row))
        raw = dict(zip(headers, padded))

        combined_meaning = _value(raw, "四縣腔華語詞義／客語舉例")
        if combined_meaning:
            zh_meaning, example_sentence = _split_advanced_meaning(
                combined_meaning
            )
        else:
            zh_meaning = _value(raw, "四縣華語詞義")
            example_sentence = _value(raw, "四縣例句")

        category_code, category = _split_category(_value(raw, "分類"))
        source_word_code = _value(raw, "編號", "編碼")
        hakka_word = _value(raw, "四縣客家語", "四縣客語詞彙")
        pinyin = _value(raw, "四縣客語標音")

        required_values = {
            "source_word_code": source_word_code,
            "hakka_word": hakka_word,
            "zh_meaning": zh_meaning,
            "pinyin": pinyin,
        }
        missing_required = [
            field for field, value in required_values.items() if not value
        ]
        placeholder = hakka_word == "此腔無此詞條"
        if missing_required or placeholder:
            flagged_rows.append(
                {
                    "row_number": row_number,
                    "reason": (
                        "dialect_placeholder"
                        if placeholder
                        else "missing_required_fields"
                    ),
                    "missing_required_fields": missing_required,
                    "raw_data": raw,
                }
            )
            continue

        example_translation = _value(raw, "四縣翻譯", "舉例華譯")
        missing_fields = [
            field
            for field, value in {
                "example_sentence": example_sentence,
                "example_translation": example_translation,
                "audio_url": "",
                "visualizable_type": "",
            }.items()
            if not value
        ]

        words.append(
            NormalizedWord(
                source_word_code=source_word_code,
                hakka_word=hakka_word,
                zh_meaning=zh_meaning,
                dialect="四縣腔",
                pinyin=pinyin,
                certification_level=level,
                category_code=category_code,
                category=category,
                subcategory="",
                part_of_speech_primary=_value(raw, "詞性1"),
                part_of_speech_secondary=_value(raw, "詞性2"),
                example_sentence=example_sentence,
                example_translation=example_translation,
                audio_url="",
                notes=_value(raw, "備註"),
                source=source_path.name,
                source_version=source_version,
                visualizable_type="",
                is_verified=True,
                normalization_status="imported",
                missing_fields=missing_fields,
                raw_data=raw,
            )
        )

    word_counts = Counter(word.hakka_word for word in words)
    duplicate_words = sorted(
        word for word, count in word_counts.items() if count > 1
    )
    for word in words:
        if word.hakka_word in duplicate_words:
            word.normalization_status = "duplicate_review"

    conflicts: list[dict[str, Any]] = []
    by_word: dict[str, list[NormalizedWord]] = {}
    for word in words:
        by_word.setdefault(word.hakka_word, []).append(word)
    for hakka_word, entries in by_word.items():
        meanings = {entry.zh_meaning for entry in entries}
        if len(meanings) > 1:
            conflicts.append(
                {
                    "hakka_word": hakka_word,
                    "source_word_codes": [
                        entry.source_word_code for entry in entries
                    ],
                    "meanings": sorted(meanings),
                }
            )
            for entry in entries:
                entry.normalization_status = "conflict_review"

    missing_counter = Counter(
        missing for word in words for missing in word.missing_fields
    )
    report = {
        "original_filename": source_path.name,
        "checksum": file_checksum(source_path),
        "sheet_name": sheet_name,
        "source_version": source_version,
        "dialect": "四縣腔",
        "certification_level": level,
        "raw_row_count": len(rows) - 1,
        "importable_count": len(words),
        "flagged_count": len(flagged_rows),
        "duplicate_word_count": len(duplicate_words),
        "duplicate_words": duplicate_words,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "missing_optional_fields": dict(sorted(missing_counter.items())),
        "flagged_rows": flagged_rows,
    }

    return ParsedVocabulary(
        path=source_path,
        checksum=report["checksum"],
        source_name="客語能力認證詞彙",
        source_version=source_version,
        sheet_name=sheet_name,
        headers=headers,
        words=words,
        flagged_rows=flagged_rows,
        report=report,
    )


def parsed_report_json(parsed: ParsedVocabulary) -> str:
    return json.dumps(parsed.report, ensure_ascii=False, indent=2)


def word_as_database_dict(word: NormalizedWord) -> dict[str, Any]:
    data = asdict(word)
    data["missing_fields_json"] = json.dumps(
        data.pop("missing_fields"),
        ensure_ascii=False,
    )
    data["raw_data_json"] = json.dumps(
        data.pop("raw_data"),
        ensure_ascii=False,
    )
    return data
