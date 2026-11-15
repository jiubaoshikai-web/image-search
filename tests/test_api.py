import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


class FakeEmbedder:
    """色名 → 色の画像を近づける決定的なテスト用エンベッダー。"""

    COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

    def embed_image(self, image):
        v = np.array(image.convert("RGB").resize((1, 1)).getpixel((0, 0)), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def embed_text(self, text):
        v = np.array(self.COLORS.get(text, (1, 1, 1)), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


def png(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path, FakeEmbedder()))


def upload(client, name, color):
    return client.post("/api/images", files=[("files", (name, png(color), "image/png"))])


def test_upload_and_search_ranks_by_similarity(client):
    for name, color in [("r.png", (255, 0, 0)), ("g.png", (0, 255, 0)), ("b.png", (0, 0, 255))]:
        assert upload(client, name, color).status_code == 200
    hits = client.get("/api/search", params={"q": "green"}).json()
    assert hits[0]["filename"] == "g.png"
    assert len(hits) == 3
    assert client.get(hits[0]["url"]).status_code == 200


def test_list_and_delete(client):
    image_id = upload(client, "r.png", (255, 0, 0)).json()["uploaded"][0]["id"]
    assert len(client.get("/api/images").json()) == 1
    assert client.delete(f"/api/images/{image_id}").status_code == 200
    assert client.get("/api/images").json() == []
    assert client.delete(f"/api/images/{image_id}").status_code == 404


def test_rejects_non_image(client):
    r = client.post("/api/images", files=[("files", ("x.txt", b"hello", "text/plain"))])
    assert r.status_code == 400


def test_empty_query_and_empty_store(client):
    assert client.get("/api/search", params={"q": "  "}).status_code == 400
    assert client.get("/api/search", params={"q": "red"}).json() == []


def test_path_traversal_blocked(client):
    assert client.get("/files/..%2Fimages.db").status_code == 404
