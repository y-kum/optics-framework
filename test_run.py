import os
import cv2
import numpy as np
from optics_framework.optics import Optics
from optics_framework.common import utils

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_run")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Strategy routing by locator type:
#   XPath / CSS   → XPathStrategy        (priority 1)
#   plain text    → TextElementStrategy  (priority 2)  DOM search
#                 → TextDetectionStrategy(priority 3)  EasyOCR  ← fallback
#   "file.png"    → ImageDetectionStrategy(priority 4) template matching
#
# Each strategy returns (x, y) pixel coordinates; Playwright does the click.
CONFIG = {
    "console": True,
    "log_level": "DEBUG",
    "execution_output_path": OUTPUT_DIR,
    "driver_sources": [
        {
            "playwright": {
                "enabled": True,
                "url": None,
                "capabilities": {
                    "browser": "chromium",
                    "headless": False,
                    "slow_mo": 500,
                    "navigation_wait_until": "networkidle",
                    "viewport": {"width": 1280, "height": 800},
                },
            }
        }
    ],
    "elements_sources": [
        {"playwright_find_element": {"enabled": True, "url": None, "capabilities": {}}},
        {"playwright_page_source":  {"enabled": True, "url": None, "capabilities": {}}},
        {"playwright_screenshot":   {"enabled": True, "url": None, "capabilities": {}}},
    ],
    "text_detection":  [{"easyocr":       {"enabled": True,  "url": None, "capabilities": {}}}],
    "image_detection": [{"templatematch": {"enabled": True,  "url": None, "capabilities": {}}}],
}

SEARCH_URL   = "https://duckduckgo.com"
SEARCH_QUERY = "Optics Framework"
SEARCH_BOX   = '//input[@name="q"]'
RESULTS_ITEM = "//article"

TEXT_SEARCH_BTN = "Search"
TEXT_RESULT_HDR = "Optics Framework"

# Templates directory — framework scans this folder at session init
TEMPLATES_DIR = os.path.join(OUTPUT_DIR, "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
TEMPLATE_FILENAME = "ddg_logo_template.png"
TEMPLATE_PATH     = os.path.join(TEMPLATES_DIR, TEMPLATE_FILENAME)


# ── helpers ───────────────────────────────────────────────────────────────────

def _crop_and_save_template(frame: np.ndarray, x: int, y: int, w: int, h: int, path: str) -> None:
    """Crop a region from a numpy BGR frame and save as PNG template."""
    crop = frame[y:y+h, x:x+w]
    cv2.imwrite(path, crop)
    print(f"  Template saved → {path}  (shape={crop.shape})")


# ── tests ─────────────────────────────────────────────────────────────────────

def search_with_xpath(query: str) -> bool:
    """Test 1 — baseline XPath flow, no OCR/vision involved."""
    optics = Optics(CONFIG)
    try:
        optics.app_management.launch_app(SEARCH_URL)
        optics.action_keyword.press_element(SEARCH_BOX)
        optics.action_keyword.enter_text_direct(query)
        optics.action_keyword.press_keycode("Enter")
        optics.action_keyword.sleep("3")
        return optics.verifier.assert_presence(RESULTS_ITEM)
    finally:
        optics.session_manager.terminate_session(optics.session_id)


def search_with_ocr_fallback(query: str) -> bool:
    """
    Test 2 — plain-text locator triggers OCR fallback.
    DOM search (TextElementStrategy) tries first; EasyOCR fires if DOM misses it.
    """
    optics = Optics(CONFIG)
    try:
        optics.app_management.launch_app(SEARCH_URL)
        optics.action_keyword.press_element(TEXT_SEARCH_BTN)   # plain text → OCR fallback
        optics.action_keyword.enter_text_direct(query)
        optics.action_keyword.press_keycode("Enter")
        optics.action_keyword.sleep("3")
        return optics.verifier.assert_presence(TEXT_RESULT_HDR)
    finally:
        optics.session_manager.terminate_session(optics.session_id)


def detect_text_on_page() -> None:
    """Test 3 — raw OCR dump: what EasyOCR sees on the DuckDuckGo homepage."""
    optics = Optics(CONFIG)
    try:
        optics.app_management.launch_app(SEARCH_URL)
        optics.action_keyword.sleep("2")

        frame = optics.verifier.strategy_manager.capture_screenshot()
        detected = optics.verifier.text_detection.detect_text(frame)

        print("\n── OCR detected text ──────────────────────")
        if detected:
            full_text, results = detected
            print(f"Full text:\n{full_text}\n")
            print("Per-word results (text | confidence):")
            for _, word, conf in results:
                print(f"  {word!r:30s}  conf={conf:.2f}")
        else:
            print("(nothing detected)")
        print("────────────────────────────────────────────\n")
    finally:
        optics.session_manager.terminate_session(optics.session_id)


def test_image_detection() -> None:
    """
    Test 4 — image / vision template matching.

    How it works:
      - Templates are discovered by scanning 'project_path' at session init.
      - They are registered by filename only (e.g. "ddg_logo_template.png").
      - Locator ".png" → ImageDetectionStrategy → SIFT+FLANN on live screenshot.

    Two sessions needed:
      Session A — captures screenshot, crops template, saves PNG to TEMPLATES_DIR.
      Session B — created with project_path=TEMPLATES_DIR so the PNG is discovered,
                  then uses the filename as the locator.
    """
    # ── Session A: capture and save the template ──────────────────────────────
    optics_a = Optics(CONFIG)
    try:
        optics_a.app_management.launch_app(SEARCH_URL)
        optics_a.action_keyword.sleep("2")

        frame = optics_a.verifier.strategy_manager.capture_screenshot()
        print(f"\n  Screenshot shape: {frame.shape}")

        # Crop the central content area (logo + search box) — this region has
        # enough SIFT keypoints (text, borders, icons) for reliable matching.
        # DuckDuckGo at 1280×800: content is centered around x=640, y=250–400.
        logo_x, logo_y, logo_w, logo_h = 280, 180, 720, 260
        _crop_and_save_template(frame, logo_x, logo_y, logo_w, logo_h, TEMPLATE_PATH)
        print(f"  Open {TEMPLATE_PATH} to verify the crop looks correct.")
    finally:
        optics_a.session_manager.terminate_session(optics_a.session_id)

    # ── Session B: discover template, then locate it on screen ────────────────
    # project_path tells the framework where to scan for template images at init
    config_with_templates = {**CONFIG, "project_path": TEMPLATES_DIR}
    optics_b = Optics(config_with_templates)
    try:
        optics_b.app_management.launch_app(SEARCH_URL)
        optics_b.action_keyword.sleep("2")

        # Locator = filename only (not full path)
        # ".png" extension → ImageDetectionStrategy → SIFT+FLANN matching
        print(f"\n  Locating element via image template: '{TEMPLATE_FILENAME}'")
        found = optics_b.verifier.assert_presence(TEMPLATE_FILENAME, timeout_str="15")
        print(f"\n  Image detection result: {'FOUND' if found else 'NOT FOUND'}")
    finally:
        optics_b.session_manager.terminate_session(optics_b.session_id)


if __name__ == "__main__":
    print("=== Test 1: XPath locators (baseline) ===")
    found = search_with_xpath(SEARCH_QUERY)
    print(f"Results found (XPath): {found}\n")

    print("=== Test 2: Plain-text locators + OCR fallback ===")
    found = search_with_ocr_fallback(SEARCH_QUERY)
    print(f"Results found (OCR fallback): {found}\n")

    print("=== Test 3: Raw OCR text dump ===")
    detect_text_on_page()

    print("=== Test 4: Image / vision template matching ===")
    test_image_detection()
