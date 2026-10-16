import pytest
from app.router import _fallback_task, load_config

def test_registry_has_only_expected_routes():
    routes = load_config()["routes"]
    assert "image_generation" in routes
    assert "coding" in routes
    assert "web_research" in routes

@pytest.mark.parametrize(("prompt", "expected"), [
    ("Create an image of a red car", "image_generation"),
    ("Generate a cinematic video of a mountain", "cinematic_video"),
    ("Debug this Python function", "coding"),
    ("Research the latest AI developments and compare sources", "web_research"),
    ("What is an API?", "general_chat"),
])
def test_fallback_classifier(prompt, expected):
    assert _fallback_task(prompt) == expected
