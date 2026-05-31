import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

router = APIRouter(prefix="/api/hakka", tags=["客語委員會API串接"])

HKTRANS_API = os.getenv("HKTRANS_API", "https://hktrans.bronci.com.tw")
HKTTS_API = os.getenv("HKTTS_API", "https://hktts.bronci.com.tw")

USERNAME = os.getenv("HAKKA_USERNAME")
PASSWORD = os.getenv("HAKKA_PASSWORD")

_cached_trans_token = None
_cached_tts_token = None


class TextRequest(BaseModel):
    text: str


async def get_trans_token(force_refresh: bool = False) -> str:
    global _cached_trans_token

    if _cached_trans_token and not force_refresh:
        return _cached_trans_token

    if not USERNAME or not PASSWORD:
        raise HTTPException(status_code=500, detail="缺少 HAKKA_USERNAME 或 HAKKA_PASSWORD")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{HKTRANS_API}/api/v1/tts/login",
            json={
                "username": USERNAME,
                "password": PASSWORD,
                "rememberMe": 1
            },
            timeout=10.0
        )

    data = response.json()

    if response.status_code != 200 or data.get("code") != 200:
        raise HTTPException(status_code=401, detail=f"客語翻譯 API 登入失敗：{data}")

    token = data.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="客語翻譯 API 未回傳 token")

    _cached_trans_token = token
    return token


async def get_tts_token(force_refresh: bool = False) -> str:
    global _cached_tts_token

    if _cached_tts_token and not force_refresh:
        return _cached_tts_token

    if not USERNAME or not PASSWORD:
        raise HTTPException(status_code=500, detail="缺少 HAKKA_USERNAME 或 HAKKA_PASSWORD")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{HKTTS_API}/api/v1/tts/login",
            json={
                "username": USERNAME,
                "password": PASSWORD,
                "rememberMe": 1
            },
            timeout=10.0
        )

    data = response.json()

    if response.status_code != 200 or data.get("code") != 200:
        raise HTTPException(status_code=401, detail=f"客語語音 API 登入失敗：{data}")

    token = data.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="客語語音 API 未回傳 token")

    _cached_tts_token = token
    return token


async def call_hakka_translate_api(endpoint: str, text: str, token: str):
    if len(text) > 300:
        raise HTTPException(status_code=400, detail="輸入字數過長，請勿超過300字")

    target_url = f"{HKTRANS_API}{endpoint}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            target_url,
            json={"input": text},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0
        )

        result = response.json()

        if response.status_code == 401 or str(result.get("code")) == "401":
            new_token = await get_trans_token(force_refresh=True)
            response = await client.post(
                target_url,
                json={"input": text},
                headers={"Authorization": f"Bearer {new_token}"},
                timeout=10.0
            )
            result = response.json()

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"客語翻譯 API 錯誤：{result}")

    return result


async def generate_hakka_tts(text: str, folder: str = "words") -> str:
    token = await get_tts_token()

    audio_dir = Path(f"static/audios/{folder}")
    audio_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.wav"
    filepath = audio_dir / filename

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{HKTTS_API}/api/v1/tts/synthesize",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "input": {
                    "text": text,
                    "textType": "characters"
                },
                "voice": {
                    "model": "broncitts",
                    "languageCode": "hak-xi-TW",
                    "name": "hak-xi-TW-vs2-F01"
                },
                "audioConfig": {
                    "speakingRate": 1.0
                },
                "outputConfig": {
                    "streamMode": 0
                }
            }
        )

        if response.status_code == 401:
            new_token = await get_tts_token(force_refresh=True)
            response = await client.post(
                f"{HKTTS_API}/api/v1/tts/synthesize",
                headers={"Authorization": f"Bearer {new_token}"},
                json={
                    "input": {
                        "text": text,
                        "textType": "characters"
                    },
                    "voice": {
                        "model": "broncitts",
                        "languageCode": "hak-xi-TW",
                        "name": "hak-xi-TW-vs2-F01"
                    },
                    "audioConfig": {
                        "speakingRate": 1.0
                    },
                    "outputConfig": {
                        "streamMode": 0
                    }
                }
            )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="客語語音合成失敗")

    with open(filepath, "wb") as f:
        f.write(response.content)

    return "/" + str(filepath).replace("\\", "/")


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def to_superscript_tone(pinyin: str) -> str:
    """把客語拼音的數字調號轉成上標數字，例如 gieu31 e31 → gieu³¹ e³¹"""
    import re
    return re.sub(r"\d+", lambda m: m.group().translate(_SUPERSCRIPT), pinyin)


@router.post("/zh-to-hakka")
async def translate_zh_to_hakka(req: TextRequest, token: str = Depends(get_trans_token)):
    return await call_hakka_translate_api(
        endpoint="/MT/translate/hakka_zh_hk",
        text=req.text,
        token=token
    )


@router.post("/hakka-to-pinyin")
async def translate_hakka_to_pinyin(req: TextRequest, token: str = Depends(get_trans_token)):
    return await call_hakka_translate_api(
        endpoint="/MT/translate/hakka_hk_py",
        text=req.text,
        token=token
    )


@router.post("/tts")
async def hakka_tts(req: TextRequest):
    audio_path = await generate_hakka_tts(req.text, folder="words")
    return {
        "code": 200,
        "audio_path": audio_path
    }