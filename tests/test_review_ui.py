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


def test_step_selection_preserves_the_sidebar_dom_and_rejects_stale_responses():
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    select_step = javascript.split(
        "async function selectStep(stepId, options = {}) {", maxsplit=1
    )[1].split("function selectionIsCurrent", maxsplit=1)[0]

    # Rebuilding the list resets its scrollTop and made every deep-step click
    # jump the reviewer back to the beginning of the episode.
    assert "renderStepList()" not in select_step
    assert "updateStepListSelection();" in select_step
    assert "const revision = ++state.selectionRevision;" in select_step
    assert select_step.count("selectionIsCurrent(") >= 2


def test_conversation_dom_is_reused_when_only_the_selected_step_changes():
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    render_conversation = javascript.split(
        "async function renderConversation(revision = state.selectionRevision) {",
        maxsplit=1,
    )[1].split("function conversationKey", maxsplit=1)[0]

    # Reusing the rendered turns keeps image layout and scroll position stable
    # when moving both forward and backward through a long episode.
    assert "state.renderedConversationKey !== key" in render_conversation
    assert "renderConversationDocument();" in render_conversation
    assert "updateConversationSelection();" in render_conversation
    assert "elements.chat.replaceChildren();" not in render_conversation


def test_step_navigation_is_recorded_in_browser_history():
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'url.searchParams.set("step", String(state.stepId));' in javascript
    assert 'window.history.pushState(null, "", url);' in javascript
    assert 'window.history.replaceState(null, "", url);' in javascript
    assert 'window.addEventListener("popstate"' in javascript
    assert "await selectStep(stepId, { historyMode: null });" in javascript
