"""Texts of the interface, one file per language: app/locales/<code>.json, {key: text}.

A text may hold values to fill in, between braces: "Speed {speed} m/s". All the languages have the same keys and
the same values to fill in (checked by the tests). Independent of Qt.
"""
import json
from pathlib import Path

LOCALES = Path(__file__).resolve().parent / "locales"
LANGUAGES = {"en": "English", "fr": "Français"}     # code: name of the language, in that language
DEFAULT = "en"


def load(code: str) -> dict:
    """The texts of the language `code`."""
    with open(LOCALES / f"{code}.json", encoding="utf-8") as f:
        return json.load(f)


def language_of(locale_name: str) -> str:
    """Language of the interface for a system locale ("fr_FR", "en_US"...): its own if known, English otherwise."""
    code = locale_name.split("_")[0].split("-")[0].lower()
    return code if code in LANGUAGES else DEFAULT


class Texts:
    """The texts of the current language: texts("key", value=...) -> text."""

    def __init__(self, code: str = DEFAULT):
        self.set_language(code)

    def set_language(self, code: str) -> None:
        self.code = code
        self._texts = load(code)

    def __call__(self, key: str, **values) -> str:
        return self._texts[key].format(**values)

    def number(self, value: float, decimals: int) -> str:
        """Number with a fixed number of decimals, with the decimal separator of the language. No minus sign before
        a number that rounds to zero ("-0.00")."""
        text = f"{value:.{decimals}f}"
        if text.startswith("-") and float(text) == 0:
            text = text[1:]
        return text.replace(".", self._texts["number.decimal_separator"])
