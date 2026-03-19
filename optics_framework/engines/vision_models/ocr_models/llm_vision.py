"""
llm_vision.py
=============
LLM-powered element locator — implements TextInterface.

Slot in the framework's text_detection fallback chain (priority 3).
When an element description like "search input box" cannot be matched by
EasyOCR as literal text, LLMVision receives the same screenshot + description,
sends it to a vision-capable LLM, and returns pixel coordinates.

Config entry::

    "text_detection": [
        {"easyocr":    {"enabled": True}},           # fast, literal-text match
        {"llm_vision": {"enabled": True, "capabilities": {
            "provider": "gemini",
            "model":    "gemini-2.5-flash-lite",
            "api_key":  "...",
        }}}
    ]

Full flow when press_element("search input box") is called
----------------------------------------------------------
  1. XPathStrategy          — skipped (not an XPath)
  2. TextElementStrategy    — DOM search for literal text, fails
  3. TextDetectionStrategy  — calls llm_vision.find_element(screenshot_np, "search input box")
     a. screenshot_np (BGR numpy) → base64 PNG
     b. Sent to Gemini/Claude/GPT with the description
     c. LLM returns {"x": N, "y": N}
     d. Returns (True, (x, y), bbox) to StrategyManager
  4. ActionKeyword sees located=(x, y) → calls driver.press_coordinates(x, y)
  5. Playwright clicks at (x, y) in the browser viewport
"""

import base64
import io
import json
import os
import re
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import BaseModel

from optics_framework.common.logging_config import internal_logger
from optics_framework.common.text_interface import TextInterface


class LLMVisionConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash-lite"
    api_key: Optional[str] = None

    class Config:
        extra = "allow"


class LLMVision(TextInterface):
    """
    LLM-powered vision engine.

    Plugs into the text_detection chain as a semantic fallback: EasyOCR handles
    literal text; LLMVision handles natural-language descriptions by visually
    understanding the screenshot.
    """

    NAME = "llm_vision"
    DEPENDENCY_TYPE = "text_detection"

    def __init__(self, config=None):
        # config is a DependencyConfig pydantic object or plain dict
        if config is None:
            caps = {}
        elif hasattr(config, "capabilities"):   # DependencyConfig object
            caps = config.capabilities or {}
        elif isinstance(config, dict):
            caps = config.get("capabilities", config)
        else:
            caps = {}

        self.cfg = LLMVisionConfig(**caps)
        self._client = self._init_client()
        internal_logger.info(
            "[LLMVision] initialized  provider=%s  model=%s",
            self.cfg.provider, self.cfg.model,
        )

    # ── Client init ───────────────────────────────────────────────────────────

    def _init_client(self):
        if self.cfg.provider == "gemini":
            from google import genai
            api_key = self.cfg.api_key or os.environ.get("GEMINI_API_KEY")
            return genai.Client(api_key=api_key)

        if self.cfg.provider == "anthropic":
            import anthropic
            api_key = self.cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
            return anthropic.Anthropic(api_key=api_key)

        if self.cfg.provider == "openai":
            import openai
            api_key = self.cfg.api_key or os.environ.get("OPENAI_API_KEY")
            return openai.OpenAI(api_key=api_key)

        raise ValueError(
            f"[LLMVision] unsupported provider '{self.cfg.provider}'. "
            "Use 'gemini', 'anthropic', or 'openai'."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _numpy_to_b64(self, frame: np.ndarray) -> str:
        """Convert a BGR numpy array to a base64-encoded PNG string."""
        rgb = frame[:, :, ::-1]
        img = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _call_llm(self, b64: str, description: str, width: int = 0, height: int = 0) -> Optional[Tuple[int, int]]:
        """Ask the LLM for the center pixel of *description*. Returns (x, y) or None."""
        dims = f"The image is exactly {width}x{height} pixels. " if width and height else ""
        prompt = (
            "You are a UI automation assistant analyzing a browser screenshot. "
            f"{dims}"
            f"Find this UI element: '{description}'. "
            "Return ONLY a JSON object with the pixel coordinates of its center "
            "in the image's own pixel space. "
            'Format: {"x": <integer>, "y": <integer>}. No explanation.'
        )
        try:
            if self.cfg.provider == "gemini":
                from google.genai import types
                img_bytes = base64.b64decode(b64)
                response = self._client.models.generate_content(
                    model=self.cfg.model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        prompt,
                    ],
                )
                raw = response.text

            elif self.cfg.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.cfg.model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/png", "data": b64,
                        }},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                raw = response.content[0].text

            else:  # openai
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    max_tokens=64,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                raw = response.choices[0].message.content

            internal_logger.info("[LLMVision] raw: %s", raw)
            match = re.search(r'\{[^}]+\}', raw)
            if not match:
                internal_logger.warning("[LLMVision] no JSON in LLM response")
                return None
            data = json.loads(match.group())
            return int(data["x"]), int(data["y"])

        except Exception as exc:
            internal_logger.error("[LLMVision] LLM call failed: %s", exc)
            return None

    # ── TextInterface ─────────────────────────────────────────────────────────

    def find_element(
        self,
        input_data: np.ndarray,
        text: str,
        index: int = None,
    ) -> Optional[Tuple[bool, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Locate the element described by *text* in the *input_data* screenshot.

        :param input_data: BGR numpy array captured by the element source.
        :param text:       Natural-language description, e.g. "search input box".
        :param index:      Ignored — LLM returns a single best-match coordinate.
        :returns:          (True, (x, y), ((x1, y1), (x2, y2))) or None if not found.
        """
        h, w = input_data.shape[:2]
        b64 = self._numpy_to_b64(input_data)
        result = self._call_llm(b64, text, width=w, height=h)
        if result is None:
            return None
        x, y = result
        if x == 0 and y == 0:
            internal_logger.warning("[LLMVision] got (0,0) — element likely not visible")
            return None
        pad = 20
        bbox = ((x - pad, y - pad), (x + pad, y + pad))
        internal_logger.info("[LLMVision] '%s' → (%d, %d)", text, x, y)
        return True, (x, y), bbox

    def element_exist(self, input_data: Any, reference_data: Any) -> Optional[Tuple[int, int]]:
        """Not used in this engine."""
        return None

    def detect_text(self, input_data: Any) -> Optional[Tuple[str, List]]:
        """
        Returns empty result so assert_elements() degrades gracefully.
        LLMVision is for find_element() (locate by description), not bulk text extraction.
        Use XPath or EasyOCR for assert_presence() verification.
        """
        return ("", [])
