"""Landing-chat actions. Deterministic so the free instance does not spend an extra model call."""

import re

# The public walkthrough on resiliohub.com/#how-it-works. One video covers the product.
DEMO_VIDEO = {
    "title": "ResilioHub product demo",
    "youtube_id": "lRMZRqRQFl8",
    "page_url": "https://resiliohub.com/#how-it-works",
}

ASK_LINE = (
    "Here is the product demo. Tap the video to play it, or open it on the walkthrough section of this page. "
    "Want me to line up a live demo with one of our experts?"
)
CALENDAR_LINE = (
    "I'm opening the calendar. Please select the date and time. "
    "This will align your live demo with our agent."
)

_LIVE = re.compile(
    r"\b(?:calendly|open (?:the )?calendar|schedule (?:a |me )?(?:a )?(?:call|meeting|demo)|"
    r"book a (?:call|meeting)|live agent|talk to (?:an |our )?expert|"
    r"align (?:a |me )?(?:a )?live demo|live demo with (?:an |our )?expert)\b",
    re.I,
)
_DEMO = re.compile(
    r"\b(?:book(?:ing)? a demo|demo video|watch (?:a |the )?demo|show (?:me )?(?:a |the )?demo|"
    r"product demo|how (?:it|this) works|walkthrough)\b",
    re.I,
)
_TOPIC = re.compile(r"\b(?:inbox|broadcasts?|campaigns?|automation|chatbots?|no-code)\b", re.I)
_YES = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|do it|book it|let'?s do it)[!.?\s]*$",
    re.I,
)


def _last_assistant(history: list) -> str:
    for item in reversed(history or []):
        role = item.role if hasattr(item, "role") else item.get("role")
        if role == "assistant":
            content = item.content if hasattr(item, "content") else item.get("content") or ""
            return content
    return ""


def plan_action(message: str, history: list, site_id: str) -> dict:
    """Return answer override, media, and action. Empty answer means keep the model reply."""
    blank = {"answer": "", "media": [], "action": None}
    if site_id != "resiliohub":
        return blank
    text = (message or "").strip()
    if _YES.match(text) and "line up a live demo" in _last_assistant(history):
        return {"answer": CALENDAR_LINE, "media": [], "action": "open_calendly"}
    if _LIVE.search(text):
        return {"answer": CALENDAR_LINE, "media": [DEMO_VIDEO], "action": "open_calendly"}
    if _DEMO.search(text):
        return {"answer": ASK_LINE, "media": [DEMO_VIDEO], "action": "ask_live_demo"}
    if _TOPIC.search(text):
        return {"answer": "", "media": [DEMO_VIDEO], "action": None}
    return blank


if __name__ == "__main__":
    assert plan_action("hey", [], "resiliohub")["action"] is None
    demo = plan_action("I want to book a demo", [], "resiliohub")
    assert demo["action"] == "ask_live_demo" and demo["media"][0]["youtube_id"] == "lRMZRqRQFl8"
    live = plan_action("talk to an expert", [], "resiliohub")
    assert live["action"] == "open_calendly"
    yes = plan_action("yes", [{"role": "assistant", "content": ASK_LINE}], "resiliohub")
    assert yes["action"] == "open_calendly"
    assert plan_action("yes", [], "resiliohub")["action"] is None
    assert plan_action("book a demo", [], "sharpquiz")["action"] is None
    topic = plan_action("How do broadcasts work?", [], "resiliohub")
    assert topic["answer"] == "" and topic["media"]
    print("actions ok")
