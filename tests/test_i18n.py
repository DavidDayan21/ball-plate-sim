"""Texts of the interface (app/i18n.py, app/locales/): the same keys and values to fill in in every language."""
import string

from app.i18n import DEFAULT, LANGUAGES, Texts, language_of, load


def fields(text: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_same_keys_and_values_to_fill_in():
    reference = load(DEFAULT)
    for code in LANGUAGES:
        texts = load(code)
        assert texts.keys() == reference.keys(), code
        for key, text in texts.items():
            assert text.strip(), (code, key)
            assert fields(text) == fields(reference[key]), (code, key)


def test_texts_and_numbers():
    tr = Texts("fr")
    assert tr("result.stabilized", time=tr.number(1.234, 2)) == "stabilisée en 1,23 s"
    tr.set_language("en")
    assert tr("result.stabilized", time=tr.number(1.234, 2)) == "stabilized in 1.23 s"


def test_language_of_the_system():
    assert language_of("fr_FR") == "fr" and language_of("fr-CA") == "fr"
    assert language_of("en_US") == "en" and language_of("he_IL") == DEFAULT and language_of("C") == DEFAULT
