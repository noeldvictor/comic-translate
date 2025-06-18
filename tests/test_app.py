import pytest

pytest.skip("GUI tests are skipped in headless environments", allow_module_level=True)

from app.ui.main_window import ComicTranslateUI


def test_comic_translate_ui_basic(qtbot):
    widget = ComicTranslateUI()
    qtbot.addWidget(widget)

    # showing the full UI can trigger platform-specific errors in headless
    # environments; we only verify that the widget initializes correctly
    assert widget.windowTitle() == "Comic Translate"
