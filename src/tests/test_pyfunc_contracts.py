"""Contract tests for the OcrPyfunc base class.

These exercise:
  * input coercion (DataFrame / list-of-dicts / single dict)
  * happy-path: subclass _infer_pages is called with decoded PIL pages
  * markdown vs json output formatting
  * per-row error isolation (one bad row doesn't kill the batch)
"""

from __future__ import annotations

import base64
import io
import json

import pandas as pd
import pytest
from PIL import Image

from doc_parser.base import OcrPyfunc


def _png_b64(color=(0, 0, 0)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeOcr(OcrPyfunc):
    HF_REPO = "fake/model"
    MODEL_NAME = "fake"
    DEFAULT_PROMPT = "fake-default"

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[int, str]] = []

    def _select_device(self) -> None:
        self.device = "cpu"

    def _load(self, weights_dir: str) -> None:
        pass

    def _infer_pages(self, pages, prompt: str) -> str:
        self.calls.append((len(pages), prompt))
        return f"PAGES={len(pages)} PROMPT={prompt}"


@pytest.fixture
def model() -> FakeOcr:
    m = FakeOcr()
    m._select_device()
    return m


class TestCoerceInput:
    def test_dataframe(self, model):
        df = pd.DataFrame([{"image_b64": _png_b64()}])
        rows = model._coerce_input(df)
        assert rows == df.to_dict(orient="records")

    def test_list_of_dicts(self, model):
        rows = model._coerce_input([{"image_b64": "abc"}, {"image_b64": "def"}])
        assert rows == [{"image_b64": "abc"}, {"image_b64": "def"}]

    def test_list_of_strings(self, model):
        rows = model._coerce_input(["abc", "def"])
        assert rows == [{"image_b64": "abc"}, {"image_b64": "def"}]

    def test_single_dict(self, model):
        rows = model._coerce_input({"image_b64": "abc"})
        assert rows == [{"image_b64": "abc"}]

    def test_unsupported_raises(self, model):
        with pytest.raises(ValueError):
            model._coerce_input(42)


class TestPredict:
    def test_default_prompt_used(self, model):
        df = pd.DataFrame([{"image_b64": _png_b64()}])
        out = model.predict(None, df)
        assert isinstance(out, pd.Series)
        assert len(out) == 1
        assert "PROMPT=fake-default" in out.iloc[0]
        assert model.calls == [(1, "fake-default")]

    def test_custom_prompt(self, model):
        df = pd.DataFrame([{"image_b64": _png_b64(), "prompt": "do the thing"}])
        out = model.predict(None, df)
        assert "PROMPT=do the thing" in out.iloc[0]

    def test_json_output_format(self, model):
        df = pd.DataFrame([{"image_b64": _png_b64(), "output_format": "json"}])
        out = model.predict(None, df)
        payload = json.loads(out.iloc[0])
        assert payload["model"] == "fake"
        assert payload["num_pages"] == 1
        assert "PAGES=1" in payload["text"]

    def test_missing_image_b64_returns_per_row_error(self, model):
        df = pd.DataFrame([{"image_b64": _png_b64()}, {"prompt": "no image"}])
        out = model.predict(None, df)
        assert len(out) == 2
        assert "PAGES=1" in out.iloc[0]
        err = json.loads(out.iloc[1])
        assert "error" in err
        assert "image_b64" in err["error"]

    def test_one_bad_row_does_not_break_batch(self, model):
        good = _png_b64()
        df = pd.DataFrame(
            [
                {"image_b64": good},
                {"image_b64": "!!!not-base64!!!"},
                {"image_b64": good},
            ]
        )
        out = model.predict(None, df)
        assert len(out) == 3
        # First and last rows should have run inference; middle row should be an error JSON.
        assert "PAGES=1" in out.iloc[0]
        assert "PAGES=1" in out.iloc[2]
