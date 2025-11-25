# Introduction

A program to align segmented Qurʾān translation text to Arabic word ranges.

# Installation

You can use either `pip` or `pipx` for installation.

If you only want to align:
```sh
pip install git+https://git.sr.ht/~rehandaphedar/rabtize
```

If you want to generate embeddings as well:
```sh
pip install "rabtize[embed] @ git+https://git.sr.ht/~rehandaphedar/rabtize"
```

If you want to use XPU:
```sh
pip install "rabtize[embed,embed-xpu] @ git+https://git.sr.ht/~rehandaphedar/rabtize" --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

A slightly modified command is required while using `pipx`:
```sh
pipx install "rabtize[embed,embed-xpu] @ git+https://git.sr.ht/~rehandaphedar/rabtize" --pip-args="--extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
```

As for other hardware acceleration backends, I cannot test them myself. However, contributions to add support for them are welcome.

# Usage

From the [Quranic Universal Library (QUL)](https://qul.tarteel.ai/resources) (or from any other source with the same schema) obtain the following:
- A [Quran script](https://qul.tarteel.ai/resources/quran-script) (`qpc-hafs-word-by-word.json`).
- A [translation](https://qul.tarteel.ai/resources/translation) (`-simple.json`) with segments in [jumlize](https://sr.ht/~rehandaphedar/jumlize) format.

These will serve as `--words`/`-w` and `--translation`/`-t` for all other commands.

Further documentation for CLI flags can be accessed by appending `-h` to the program or to any subcommand.

## Generating Embeddings

Generate 2 sets of embeddings:

```sh
rabtize embed spans embeddings/spans.npz
rabtize embed segments embeddings/en-sahih-international-simple.npz
```

1. Spans embeddings: These are the embeddings for each possible (sequential) range of words found in the Quran script. Currently, these amount to just under 700k texts. These embeddings only have to be generated once (per model) and can be reused across different translations.
2. Segments embeddings: These are the embeddings for each segment in the translation. These must be generated separately for each translation.


## Aligning Segments

```sh
rabtize align -sp embeddings/spans.npz -se embeddings/en-sahih-international-simple.npz results/en-sahih-international-simple.json
```

# Output Format

The resulting output will be of the following format (Only the `words` field is added, rest of the output is the same as [jumlize](https://sr.ht/~rehandaphedar/jumlize/)'s output format):

```json
{
	"[verse_key]": {
		"t": "[text]",
		"segments": [
			{
				"t": "[segment_text]",
				"words": {
					"start": [start_index],
					"end": [end_index]
				}
			},
			{
				"t": "[segment_text]",
				"words": {
					"start": [start_index],
					"end": [end_index]
				}
			}
		]
	}
}
```

Both `start_index` and `end_index` are 0 based. `start_index` is inclusive while `end_index` is exclusive.

# Results

Aligned translations can be found under `results/`.
