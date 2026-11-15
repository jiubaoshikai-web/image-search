# 画像検索アプリ

手持ちの画像をアップロードし、**文章（日本語・英語）で検索**できるアプリです。
CLIP でテキストと画像を同じベクトル空間に埋め込み、コサイン類似度で並べ替えます。

## セットアップ

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app_factory --factory --reload
```

http://localhost:8000 を開き、「画像を追加」でアップロード → 検索ボックスに「海辺の夕日」などと入力。
初回はモデル（数百MB）のダウンロードが走ります。

## 構成

- `app/embedder.py` — CLIP エンベッダー（差し替え可能）
- `app/store.py` — SQLite にメタデータ＋ベクトル保存、numpy で全件検索
- `app/main.py` — FastAPI (`/api/images`, `/api/search`)
- `app/static/index.html` — 簡易 Web UI
- データは `data/`（`IMAGE_SEARCH_DATA` で変更可）

## テスト

```bash
pytest
```
