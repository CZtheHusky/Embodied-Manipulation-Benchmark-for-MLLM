from __future__ import annotations

import re

from mllm_teleop_benchmark.review.server import STATIC_DIR


def _rule(source: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]+)\}}", source)
    assert match, f"Missing CSS rule for {selector}"
    return match.group(1)


def test_desktop_review_bounds_each_column_to_the_viewport():
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "height: 100dvh;" in _rule(css, ".app")
    assert "overflow: hidden;" in _rule(css, ".workspace")
    assert "overflow-y: auto;" in _rule(css, ".step-list")
    assert "overflow-y: auto;" in _rule(css, ".chat")
    assert "overflow: auto;" in _rule(css, ".json-view")
    assert css.count("overscroll-behavior: contain;") == 3


def test_narrow_review_restores_document_scrolling():
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    narrow = css.split("@media (max-width: 980px)", maxsplit=1)[1]
    narrow = narrow.split("@media (max-width: 640px)", maxsplit=1)[0]

    assert "overflow-y: auto;" in _rule(narrow, "html,\n  body")
    assert "height: auto;" in _rule(narrow, ".app")
    assert "overflow: visible;" in _rule(narrow, ".workspace")
    assert "grid-template-rows: auto auto auto;" in _rule(narrow, ".main")
    assert "overflow-y: visible;" in _rule(narrow, ".chat")


def test_conversation_selection_only_scrolls_the_chat_container():
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "imageNode.width = image.width;" in javascript
    assert "imageNode.height = image.height;" in javascript
    assert "elements.chat.scrollTo({" in javascript
    assert "elements.chat.scrollTop + selectedBounds.top - chatBounds.top" in javascript
    assert "scrollIntoView" not in javascript
