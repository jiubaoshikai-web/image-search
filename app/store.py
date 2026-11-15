"""SQLite に画像メタデータと埋め込みベクトルを保存する。"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np


class ImageStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )"""
        )
        self._conn.commit()

    def add(self, filename: str, stored_name: str, embedding: np.ndarray) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO images (filename, stored_name, embedding) VALUES (?, ?, ?)",
                (filename, stored_name, embedding.astype(np.float32).tobytes()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, filename, stored_name FROM images ORDER BY id DESC"
            ).fetchall()
        return [{"id": r[0], "filename": r[1], "stored_name": r[2]} for r in rows]

    def get(self, image_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT id, filename, stored_name FROM images WHERE id = ?", (image_id,)
            ).fetchone()
        return {"id": r[0], "filename": r[1], "stored_name": r[2]} if r else None

    def delete(self, image_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
            self._conn.commit()

    def search(self, query: np.ndarray, limit: int = 20) -> list[dict]:
        """コサイン類似度 (正規化済みベクトルの内積) の降順で返す。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, filename, stored_name, embedding FROM images"
            ).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        scores = matrix @ query.astype(np.float32)
        order = np.argsort(-scores)[:limit]
        return [
            {
                "id": rows[i][0],
                "filename": rows[i][1],
                "stored_name": rows[i][2],
                "score": float(scores[i]),
            }
            for i in order
        ]
