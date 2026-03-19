# LLM Vision — Element Detection Guide

LLM Vision is a semantic fallback engine in the `text_detection` chain. Instead of matching literal text or XPath, it sends a browser screenshot to a vision-capable LLM (Gemini, Claude, or OpenAI) and asks it to locate an element by natural-language description.

**When to use it:** when the element has no stable XPath and no visible text — e.g. icon buttons, placeholder fields, visual landmarks.

---

## Installation

```bash
# Core dependencies
poetry add google-genai Pillow

# Playwright browser driver + binary
poetry add playwright
playwright install chromium
```

## API Key Setup

The API key must be supplied via an environment variable — **never hard-code it in source files**.

### Set the environment variable

**Linux / macOS**

```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Windows (Command Prompt)**

```cmd
set GEMINI_API_KEY=your_api_key_here
```

**Windows (PowerShell)**

```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

---

## Configuration

Add `llm_vision` to the `text_detection` list in your Optics config:

```python
CONFIG = {
    "driver_sources": [
        {
            "playwright": {
                "enabled": True,
                "url": None,
                "capabilities": {
                    "browser": "chromium",
                    "headless": False,
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
    "text_detection": [
        {"llm_vision": {"enabled": True, "url": None, "capabilities": {
            "provider": "gemini",           # gemini | anthropic | openai
            "model":    "gemini-2.0-flash",
            "api_key":  None,               # or set via GEMINI_API_KEY env var
        }}}
    ],
}
```

### Supported providers

| Provider    | Env var             | Example model              |
| ----------- | ------------------- | -------------------------- |
| `gemini`    | `GEMINI_API_KEY`    | `gemini-2.0-flash`         |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-latest` |
| `openai`    | `OPENAI_API_KEY`    | `gpt-4o`                   |

> Prefer setting the API key via environment variable rather than hardcoding it in config.

---

## Usage

```python
from optics_framework.optics import Optics

optics = Optics(CONFIG)
optics.app_management.launch_app("https://example.com")
optics.action_keyword.sleep("3")  # let the page fully render

# Describe the element in plain English
optics.action_keyword.press_element(
    "search input field with placeholder text 'Search privately'"
)

optics.action_keyword.enter_text_direct("my query")
optics.action_keyword.press_keycode("Enter")
```

See [`test_run_llm_driver.py`](../test_run_llm_driver.py) for a full working example.

---

## How it works

1. `press_element("description")` runs through the strategy chain
2. XPath and DOM strategies are tried first (fast, zero LLM cost)
3. If both fail, `LLMVision.find_element(screenshot, description)` is called
4. The screenshot (numpy BGR array) is base64-encoded and sent to the LLM
5. The LLM returns `{"x": N, "y": N}` — the center pixel of the element
6. Playwright clicks at those coordinates

---

## Tips

- **Be specific in descriptions.** "search input" is weaker than "search input field with placeholder text 'Search privately'".
- **Allow render time.** Add a `sleep("3")` after page load before calling `press_element`, so the LLM sees a fully rendered page.
- **Use LLM Vision as a fallback.** Keep XPath or DOM strategies enabled alongside it — they are faster and cheaper.
