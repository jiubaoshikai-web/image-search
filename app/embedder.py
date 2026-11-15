"""画像・テキストを同じベクトル空間に埋め込むエンベッダー。"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    def embed_image(self, image: Image.Image) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


class ClipEmbedder:
    """CLIP (画像) + 多言語CLIPテキストエンコーダ (日本語クエリ対応)。

    初回利用時にモデルをダウンロードする (数百MB)。
    """

    IMAGE_MODEL = "clip-ViT-B-32"
    TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def __init__(self) -> None:
        self._image_model = None
        self._text_model = None

    def _load_image(self):
        if self._image_model is None:
            from sentence_transformers import SentenceTransformer

            self._image_model = SentenceTransformer(self.IMAGE_MODEL)
        return self._image_model

    def _load_text(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(self.TEXT_MODEL)
        return self._text_model

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return _normalize(self._load_image().encode(image.convert("RGB")))

    def embed_text(self, text: str) -> np.ndarray:
        return _normalize(self._load_text().encode(text))
