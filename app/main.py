from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .embedder import ClipEmbedder, Embedder
from .store import ImageStore

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def create_app(data_dir: Path | None = None, embedder: Embedder | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("IMAGE_SEARCH_DATA", "data"))
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Image Search")
    app.state.store = ImageStore(data_dir / "images.db")
    app.state.embedder = embedder or ClipEmbedder()
    app.state.upload_dir = upload_dir

    def get_state(request: Request):
        return request.app.state

    @app.post("/api/images")
    async def upload(files: list[UploadFile] = File(...), state=Depends(get_state)):
        results = []
        for f in files:
            raw = await f.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{f.filename} が大きすぎます (20MBまで)")
            try:
                image = Image.open(io.BytesIO(raw))
                image.load()
            except (UnidentifiedImageError, OSError):
                raise HTTPException(400, f"{f.filename} は画像として読み込めません")
            stored = f"{uuid.uuid4().hex}.png"
            image.convert("RGB").save(state.upload_dir / stored, format="PNG")
            image_id = state.store.add(
                f.filename or stored, stored, state.embedder.embed_image(image)
            )
            results.append({"id": image_id, "filename": f.filename})
        return {"uploaded": results}

    @app.get("/api/images")
    def list_images(state=Depends(get_state)):
        return [
            {**i, "url": f"/files/{i['stored_name']}"} for i in state.store.list()
        ]

    @app.delete("/api/images/{image_id}")
    def delete_image(image_id: int, state=Depends(get_state)):
        item = state.store.get(image_id)
        if not item:
            raise HTTPException(404, "画像が見つかりません")
        state.store.delete(image_id)
        (state.upload_dir / item["stored_name"]).unlink(missing_ok=True)
        return {"deleted": image_id}

    @app.get("/api/search")
    def search(q: str, limit: int = 20, state=Depends(get_state)):
        q = q.strip()
        if not q:
            raise HTTPException(400, "検索ワードを入力してください")
        hits = state.store.search(state.embedder.embed_text(q), limit=min(limit, 100))
        return [
            {
                "id": h["id"],
                "filename": h["filename"],
                "score": round(h["score"], 4),
                "url": f"/files/{h['stored_name']}",
            }
            for h in hits
        ]

    @app.get("/files/{name}")
    def get_file(name: str, state=Depends(get_state)):
        path = (state.upload_dir / name).resolve()
        if path.parent != state.upload_dir.resolve() or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/")
    def index():
        return FileResponse(BASE_DIR / "static" / "index.html")

    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    return app


def app_factory() -> FastAPI:
    return create_app()
