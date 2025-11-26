import argparse
import logging
import sys
import msgspec
import numpy as np

from rabtize.util import (
    Words,
    Translation,
    TranslationWords,
    generate_spans,
    generate_verses,
)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    _ = parser.add_argument(
        "--words",
        "-w",
        type=str,
        default="qpc-hafs-word-by-word.json",
        help="Path to JSON file containing Quran script data (word by word) in QUL format.",
    )
    _ = parser.add_argument(
        "--translation",
        "-t",
        type=str,
        default="results/en-sahih-international-simple.json",
        help="Path to JSON file containing Quran translation data in QUL format, with segments in jumlize format (will be overwritten).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    args_embed = subparsers.add_parser(
        "embed", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _ = args_embed.add_argument(
        "type",
        type=str,
        choices=["spans", "segments"],
        help="What to embed. `spans` embeds all possible word combinations. `segments` embeds translation text segments.",
    )
    _ = args_embed.add_argument(
        "output",
        type=str,
        default="embeddings/spans.npz",
    )
    _ = args_embed.add_argument(
        "--model",
        "-m",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="-",
    )
    _ = args_embed.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="-",
    )
    _ = args_embed.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=512,
        help="-",
    )
    args_embed.set_defaults(func=embed)

    args_align = subparsers.add_parser(
        "align", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _ = args_align.add_argument(
        "output",
        type=str,
    )
    _ = args_align.add_argument(
        "--spans",
        "-sp",
        type=str,
        default="embeddings/spans.npz",
        help="-",
    )

    _ = args_align.add_argument(
        "--segments",
        "-se",
        type=str,
        default="embeddings/en-sahih-international-simple.npz",
        help="-",
    )
    args_align.set_defaults(func=align)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Running with configuration {args}")

    logging.info("Loading words and translation...")
    global translation
    with open(args.words) as f:
        words = msgspec.json.decode(f.read(), type=Words)
    with open(args.translation) as f:
        translation = msgspec.json.decode(f.read(), type=Translation)

    logging.info("Normalising words and translation...")
    global verses, spans, words_by_verse_key, segments_by_verse_key
    verses = generate_verses(words, translation)
    spans = generate_spans(verses)
    words_by_verse_key = {verse_key: verse.words for verse_key, verse in verses.items()}
    segments_by_verse_key = {
        verse_key: verse.segments for verse_key, verse in verses.items()
    }

    args.func(args)


def embed(args: argparse.Namespace):
    from sentence_transformers import SentenceTransformer

    logging.info("Loading model...")
    model = SentenceTransformer(args.model, device=args.device)

    match args.type:
        case "spans":
            texts = list(spans.keys())

            logging.info("Generating embeddings...")
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=args.batch_size,
            )

            logging.info("Saving embeddings...")
            np.savez(
                args.output,
                model=args.model,
                words=words_by_verse_key,
                embeddings=embeddings,
            )
        case "segments":
            texts = [
                segment
                for verse in verses.values()
                if len(verse.segments) > 1
                for segment in verse.segments
            ]

            logging.info("Generating embeddings...")
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=args.batch_size,
            )

            logging.info("Saving embeddings...")
            np.savez(
                args.output,
                model=args.model,
                segments=segments_by_verse_key,
                embeddings=embeddings,
            )


def align(args: argparse.Namespace):
    logging.info("Loading spans embeddings...")
    spans_cache = np.load(args.spans, allow_pickle=True)
    logging.info("Loading segments embeddings...")
    segments_cache = np.load(args.segments, allow_pickle=True)

    spans_model = spans_cache["model"].item()
    spans_embeddings = spans_cache["embeddings"]
    cached_words = spans_cache["words"]
    segments_model = segments_cache["model"].item()
    segments_embeddings = segments_cache["embeddings"]
    cached_segments = segments_cache["segments"]

    span_embeddings_dimension = spans_embeddings.shape[1]

    logging.info("Verifying loaded data...")
    if spans_model != segments_model:
        logging.critical(
            "Mismatch between models used in the spans cache and the segments cache."
        )
        sys.exit(1)
    if cached_words != words_by_verse_key:
        logging.critical("Mismatch between words in the JSON and the spans cache.")
        sys.exit(1)
    if cached_segments != segments_by_verse_key:
        logging.critical(
            "Mismatch between segments in the JSON and the segments cache."
        )
        sys.exit(1)

    logging.info("Arranging spans embeddings by verse...")
    for span_text, embedding in zip(spans.keys(), spans_embeddings):
        for verse_key, i, j in spans[span_text]:
            verses[verse_key].spans_embeddings[(i, j)] = embedding

    logging.info(
        "Arranging segments embeddings by verse, and spans embeddings into matrices..."
    )
    marker = 0
    for verse_key, verse in verses.items():
        logging.info(f"Processing verse {verse_key}")

        num_words = len(verse.words)
        num_segments = len(verse.segments)

        if num_segments == 1:
            alignments = [{"start": 0, "end": num_words}]
        else:
            spans_matrix = np.full(
                (num_words, num_words, span_embeddings_dimension), np.nan
            )
            for (i, j), embedding in verse.spans_embeddings.items():
                spans_matrix[i, j] = embedding

            alignments = process_verse(
                verse.words,
                verse.segments,
                spans_matrix,
                segments_embeddings[marker : marker + num_segments],
            )
            marker += num_segments

        for idx, alignment in enumerate(alignments):
            translation[verse_key].segments[idx].words = alignment

    logging.info("Writing output to file...")
    with open(args.output, "wb") as f:
        _ = f.write(msgspec.json.encode(translation))


def process_verse(
    words: list[str],
    segments: list[str],
    spans_matrix: np.ndarray,
    segments_embeddings: np.ndarray,
):
    num_words = len(words)
    num_segments = len(segments)

    default_segments = [TranslationWords(start=0, end=num_words)]

    if num_segments == 1:
        return default_segments

    path = {}
    dp = np.full((num_segments + 1, num_words + 1), -np.inf)
    dp[0, 0] = 0

    for segment_number in range(1, num_segments + 1):
        segment_embeddings = segments_embeddings[segment_number - 1]
        for span_end_number in range(1, num_words + 1):
            for span_start_number in range(span_end_number):
                span_embeddings = spans_matrix[span_start_number, span_end_number - 1]
                if np.isnan(span_embeddings[0]):
                    continue

                similarity = np.dot(segment_embeddings, span_embeddings)
                current_score = dp[segment_number - 1, span_start_number] + similarity

                if current_score > dp[segment_number, span_end_number]:
                    dp[segment_number, span_end_number] = current_score
                    path[(segment_number, span_end_number)] = span_start_number

    if dp[num_segments, num_words] == -np.inf:
        return default_segments

    alignments: list[TranslationWords] = []
    span_end_number = num_words
    for segment_number in range(num_segments, 0, -1):
        span_start_number = path.get((segment_number, span_end_number), 0)
        alignments.append(
            TranslationWords(start=span_start_number, end=span_end_number)
        )
        span_end_number = span_start_number
    alignments.reverse()

    return alignments


if __name__ == "__main__":
    main()
