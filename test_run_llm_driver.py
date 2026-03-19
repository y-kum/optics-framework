import os
from optics_framework.optics import Optics

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_run")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEARCH_URL   = "https://duckduckgo.com"
SEARCH_QUERY = "Optics Framework"

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
    # LLM Vision as the text_detection engine.
    # Natural-language element descriptions route here after XPath / DOM strategies fail.
    "text_detection": [
        {"llm_vision": {"enabled": True, "url": None, "capabilities": {
            "provider": "gemini",
            "model":    "gemini-2.5-flash",
            "api_key":  "AIzaSyB_brTQNXKNbuAtz5XMMsRfh-vqfnrT6SU",
        }}}
    ],
    "image_detection": [{"templatematch": {"enabled": False, "url": None, "capabilities": {}}}],
}


def search_with_llm_vision(query: str) -> bool:
    """
    Open DuckDuckGo, use LLM Vision to locate the search box, type the query,
    press Enter, then verify search results appeared via XPath.
    """
    optics = Optics(CONFIG)
    try:
        # Step 1: launch browser and let the page fully settle
        optics.app_management.launch_app(SEARCH_URL)
        optics.action_keyword.sleep("4")   # allow JS/CSS to finish rendering

        # Step 2: LLM Vision locates the search box from a Playwright screenshot.
        # The description is intentionally specific so the LLM picks the large
        # central search field — not the small navigation icons in the header.
        optics.action_keyword.press_element(
            "search input field with placeholder text 'Search privately'"
        )

        # Step 3: type the query (Playwright keyboard — no element location needed)
        optics.action_keyword.enter_text_direct(query)

        # Step 4: submit and wait for results
        optics.action_keyword.press_keycode("Enter")
        optics.action_keyword.sleep("3")

        # Step 5: verify results via XPath (Playwright — fast and reliable)
        return optics.verifier.assert_presence("//article")

    finally:
        optics.session_manager.terminate_session(optics.session_id)


if __name__ == "__main__":
    print("=== LLM Vision element detection ===")
    found = search_with_llm_vision(SEARCH_QUERY)
    print(f"Search results found: {found}")
