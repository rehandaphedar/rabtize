from collections import defaultdict

from pyarabic import araby


def generate_verses(words: dict, translation: dict) -> dict[str, dict]:
    verses = defaultdict(lambda: {"words": [], "segments": [], "spans_embeddings": {}})

    for word in sorted(words.values(), key=sort_word):
        verse_key = f"{word['surah']}:{word['ayah']}"
        verses[verse_key]["words"].append(normalize_word(word["text"]))
    for verse_key in verses:
        verses[verse_key]["words"].pop()

    for verse_key, verse in sorted(translation.items(), key=sort_verse):
        verses[verse_key]["segments"] = [segment["t"] for segment in verse["segments"]]

    return dict(verses)


def generate_spans(verses: dict[str, dict]) -> dict[str, list[tuple[str, int, int]]]:
    spans = defaultdict(list)
    for verse_key, verse in verses.items():
        num_words = len(verse["words"])
        for i in range(num_words):
            for j in range(i, num_words):
                span_text = " ".join(verse["words"][i : j + 1])
                spans[span_text].append((verse_key, i, j))

    return dict(spans)


def sort_word(word):
    return int(word["surah"]), int(word["ayah"]), int(word["word"])


def sort_verse(item):
    verse_key = item[0]
    chapter_number, verse_number = verse_key.split(":")
    return int(chapter_number), int(verse_number)


def normalize_word(word: str) -> str:
    word = araby.strip_tashkeel(word)
    word = "".join(char for char in word if char in araby.LETTERS)
    return word
