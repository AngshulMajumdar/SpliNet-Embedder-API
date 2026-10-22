from pathlib import Path
from contextlib import asynccontextmanager
import hashlib
import json
import os
import unicodedata
import urllib.request
from typing import List

import numpy as np
import sentencepiece as spm
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# SpliNet Embedder API
# Model assets are distributed independently through
# GitHub Release v1.0.0 of this repository.
# ============================================================

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"

RELEASE_TAG = "v1.0.0"
RELEASE_BASE = (
    "https://github.com/AngshulMajumdar/"
    "SpliNet-Embedder-API/releases/download/"
    f"{RELEASE_TAG}"
)

ASSETS = {
    "splinet_embedder.ts": (
        "85a9b43d07420329c52d9ccd37b63fd"
        "af0f2f5d86d4a265c48fb467cb655fff0"
    ),
    "spiece.model": (
        "119ec6b2af9cbbc56f297bd606b69f79"
        "f6e3130a34ee56122ee813a58d15bb9d"
    ),
    "model_info.json": (
        "ac63e9fd9385332471bed35b68a7a902"
        "754c424806f7a7064a6b8e5da04cff1c"
    ),
}

PAD_ID = 3
CLS_ID = 4
SEP_ID = 5
MAX_LENGTH = 512
EMBEDDING_DIM = 768

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DEFAULT_BATCH = int(
    os.environ.get(
        "SPLINET_BATCH_SIZE",
        "256" if DEVICE.type == "cuda" else "8"
    )
)

_MODEL = None
_SP = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download_asset(filename: str, expected_sha256: str) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    destination = ASSET_DIR / filename

    if destination.exists():
        if _sha256(destination) == expected_sha256:
            return destination

        print(f"Checksum mismatch for existing {filename}; re-downloading.")
        destination.unlink()

    temp = ASSET_DIR / f"{filename}.download"
    if temp.exists():
        temp.unlink()

    url = f"{RELEASE_BASE}/{filename}"

    print(f"Downloading {filename}")
    print(f"  {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SpliNet-Embedder-API/1.0.0"}
    )

    try:
        with urllib.request.urlopen(request) as response, temp.open("wb") as f:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0

            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break

                f.write(block)
                downloaded += len(block)

                if total:
                    pct = 100.0 * downloaded / total
                    print(
                        f"\r  {downloaded / 1024**2:.1f} / "
                        f"{total / 1024**2:.1f} MiB "
                        f"({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

            if total:
                print()

    except Exception:
        if temp.exists():
            temp.unlink()
        raise

    actual_sha256 = _sha256(temp)

    if actual_sha256 != expected_sha256:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 verification failed for {filename}.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}"
        )

    temp.replace(destination)

    print(
        f"Verified {filename} "
        f"({destination.stat().st_size / 1024**2:.2f} MiB)"
    )

    return destination


def ensure_assets() -> None:
    for filename, sha256 in ASSETS.items():
        _download_asset(filename, sha256)


def _load_runtime() -> None:
    global _MODEL, _SP

    if _MODEL is not None and _SP is not None:
        return

    ensure_assets()

    print(f"Loading SpliNet Embedder on {DEVICE}...")

    _MODEL = torch.jit.load(
        str(ASSET_DIR / "splinet_embedder.ts"),
        map_location=DEVICE,
    ).eval()

    _SP = spm.SentencePieceProcessor(
        model_file=str(ASSET_DIR / "spiece.model")
    )

    print("SpliNet Embedder ready.")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(text.split())


def _tokenize(texts: List[str]):
    _load_runtime()

    ids = torch.full(
        (len(texts), MAX_LENGTH),
        PAD_ID,
        dtype=torch.long,
    )

    mask = torch.zeros(
        (len(texts), MAX_LENGTH),
        dtype=torch.long,
    )

    for i, text in enumerate(texts):
        pieces = list(
            _SP.encode(
                _normalize(text),
                out_type=int,
            )
        )[: MAX_LENGTH - 2]

        seq = [CLS_ID] + pieces + [SEP_ID]
        n = len(seq)

        ids[i, :n] = torch.tensor(seq, dtype=torch.long)
        mask[i, :n] = 1

    return ids, mask


def encode(
    texts: List[str],
    batch_size: int | None = None,
) -> np.ndarray:
    _load_runtime()

    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return np.empty(
            (0, EMBEDDING_DIM),
            dtype=np.float32,
        )

    bs = max(
        1,
        int(batch_size or DEFAULT_BATCH),
    )

    outputs = []
    start = 0

    while start < len(texts):
        current = min(
            bs,
            len(texts) - start,
        )

        try:
            ids, mask = _tokenize(
                texts[start:start + current]
            )

            ids = ids.to(DEVICE)
            mask = mask.to(DEVICE)

            with torch.inference_mode():
                if DEVICE.type == "cuda":
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        z = _MODEL(ids, mask)
                else:
                    z = _MODEL(ids, mask)

            outputs.append(
                z.float().cpu().numpy()
            )

            start += current

        except torch.OutOfMemoryError:
            if DEVICE.type != "cuda" or current == 1:
                raise

            torch.cuda.empty_cache()
            bs = max(1, current // 2)

    return np.concatenate(
        outputs,
        axis=0,
    ).astype(np.float32)


def similarity(text_a: str, text_b: str) -> float:
    z = encode([text_a, text_b])
    return float(np.dot(z[0], z[1]))


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1)
    batch_size: int | None = Field(default=None, ge=1)


class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_runtime()
    yield


app = FastAPI(
    title="SpliNet Embedder API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "release": RELEASE_TAG,
        "embedding_dimension": EMBEDDING_DIM,
    }


@app.get("/model-info")
def model_info():
    _load_runtime()

    with (ASSET_DIR / "model_info.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@app.post("/embed")
def embed_endpoint(req: EmbedRequest):
    try:
        z = encode(
            req.texts,
            req.batch_size,
        )

        return {
            "dimension": EMBEDDING_DIM,
            "count": len(req.texts),
            "embeddings": z.tolist(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/similarity")
def similarity_endpoint(req: SimilarityRequest):
    try:
        return {
            "cosine_similarity": similarity(
                req.text_a,
                req.text_b,
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
