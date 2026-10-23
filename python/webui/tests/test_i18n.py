# Checks on the UI string tables (static/i18n.js).
#
# The strings themselves are data, but a few things about them can break
# silently and are worth a test: a key used by the markup or by app.js
# with no entry in the tables, a language that is missing a key the
# other one has, Japanese text in the English UI, and the default
# language drifting away from English. All of these show up on screen
# without any error.
#
# The tables are JavaScript, so they are parsed here with a small regex
# rather than imported. That is enough for "which keys exist".

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(os.path.dirname(HERE), "static")

# Hiragana, katakana, CJK ideographs and full-width forms.
JAPANESE = re.compile("[%s-%s%s-%s%s-%s%s-%s]" % tuple(map(chr, (
    0x3040, 0x30FF, 0x3400, 0x4DBF, 0x4E00, 0x9FFF, 0xFF00, 0xFFEF))))


def read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as f:
        return f.read()


def table_blocks():
    """Return {lang: source text of its table} parsed from i18n.js."""
    src = read("i18n.js")
    body = src[src.index("const STRINGS"):src.index("const I18N")]
    return {m.group(1): m.group(2)
            for m in re.finditer(r"^  (\w+): \{(.*?)^  \},", body, re.S | re.M)}


def tables():
    """Return {lang: set(keys)} parsed from i18n.js."""
    return {lang: set(re.findall(r'"([\w.]+)":', block))
            for lang, block in table_blocks().items()}


def used_keys():
    """Keys referenced from the markup and from app.js."""
    html = read("index.html")
    js = read("app.js")
    keys = set(re.findall(r'data-i18n(?:-ph)?="([\w.]+)"', html))
    keys |= set(re.findall(r't\("([\w.]+)"', js))
    return keys


def test_languages_present():
    langs = tables()
    assert "en" in langs and "ja" in langs
    assert len(langs["en"]) > 20


def test_every_language_has_the_same_keys():
    langs = tables()
    en, ja = langs["en"], langs["ja"]
    assert en - ja == set(), "missing from ja: %s" % sorted(en - ja)
    assert ja - en == set(), "missing from en: %s" % sorted(ja - en)


def test_used_keys_are_defined():
    langs = tables()
    missing = used_keys() - langs["en"]
    assert missing == set(), "used but not defined: %s" % sorted(missing)


def test_english_table_has_no_japanese_text():
    # Catches an entry copied from the ja table and left untranslated.
    bad = [line.strip() for line in table_blocks()["en"].splitlines()
           if JAPANESE.search(line)]
    assert bad == [], "japanese text left in the en table: %s" % bad


def test_default_language_is_english():
    src = read("i18n.js")
    i18n = src[src.index("const I18N"):]
    assert re.search(r'^  lang: "en",$', i18n, re.M), "I18N.lang is not en"
    assert "STRINGS[this.lang] || STRINGS.en" in i18n
    html = read("index.html")
    assert '<html lang="en">' in html
    selector = html[html.index('<select id="lang">'):]
    first = re.search(r'<option value="(\w+)"', selector).group(1)
    assert first == "en", "the language selector starts with %s" % first


def test_markup_text_is_english():
    # The text in index.html is what shows before app.js runs; only the
    # name of the Japanese language in the selector may be Japanese.
    bad = [line.strip() for line in read("index.html").splitlines()
           if JAPANESE.search(line) and '<option value="ja">' not in line]
    assert bad == [], "japanese text in index.html: %s" % bad


def test_no_hardcoded_japanese_left_in_app_js():
    # Every user-visible string should come from the tables.
    js = read("app.js")
    lines = [l for l in js.splitlines()
             if JAPANESE.search(l) and not l.strip().startswith("//")]
    assert lines == [], "hardcoded text in app.js: %s" % lines
