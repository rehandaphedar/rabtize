from collections import defaultdict
import msgspec

import numpy as np
from pyarabic import araby


class Verse(msgspec.Struct):
    words: list[str] = msgspec.field(default_factory=list)
    segments: list[str] = msgspec.field(default_factory=list)
    spans_embeddings: dict[tuple[int, int], np.ndarray] = msgspec.field(
        default_factory=dict
    )


class Word(msgspec.Struct):
    id: int
    surah: str
    ayah: str
    word: str
    location: str
    text: str


class TranslationWords(msgspec.Struct):
    start: int
    end: int


class TranslationSegment(msgspec.Struct):
    t: str
    words: TranslationWords


class TranslationVerse(msgspec.Struct):
    t: str
    segments: list[TranslationSegment]


Words = dict[str, Word]
Translation = dict[str, TranslationVerse]
Verses = dict[str, Verse]
Span = list[tuple[str, int, int]]
Spans = dict[str, Span]


def generate_verses(words: Words, translation: Translation):
    verses: Verses = defaultdict(Verse)

    for word in sorted(words.values(), key=sort_word):
        verse_key = f"{word.surah}:{word.ayah}"
        verses[verse_key].words.append(normalize_word(word.text))
    for verse_key in verses:
        _ = verses[verse_key].words.pop()

    for verse_key, verse in sorted(translation.items(), key=sort_verse):
        verses[verse_key].segments = [segment.t for segment in verse.segments]

    return dict(verses)


def generate_spans(verses: Verses):
    spans: Spans = defaultdict(Span)
    for verse_key, verse in verses.items():
        num_words = len(verse.words)
        for i in range(num_words):
            for j in range(i, num_words):
                span_text = " ".join(verse.words[i : j + 1])
                spans[span_text].append((verse_key, i, j))

    return dict(spans)


def sort_word(word: Word):
    return int(word.surah), int(word.ayah), int(word.word)


def sort_verse(item: tuple[str, TranslationVerse]):
    verse_key = item[0]
    chapter_number, verse_number = verse_key.split(":")
    return int(chapter_number), int(verse_number)


def normalize_word(word: str) -> str:
    word = araby.strip_tashkeel(word)
    word = "".join(char for char in word if char in araby.LETTERS)
    return word
