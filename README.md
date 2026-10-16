# AuditScribe

**Autoregressive smart-contract audit prose generation with Simple RNN, Bidirectional RNN, LSTM and GRU.**

AuditScribe compares four recurrent neural networks on next-token prediction and sequential text generation from public Web3 audit prose. It is an educational experiment for Foundations of Generative AI (ICT4403), Assignment 2. It generates audit-style continuations; it does not analyze contracts or verify vulnerabilities.

## Dataset and provenance

Source: [Solodit Content](https://github.com/solodit/solodit_content), pinned to [`4d5330d12978c7a8fc2aab5b171112784fb671bd`](https://github.com/solodit/solodit_content/tree/4d5330d12978c7a8fc2aab5b171112784fb671bd).

The prepared corpus contains 483 reports and 5,140 cleaned passages. Text is drawn from finding descriptions, impacts and recommendations, with other finding prose retained. The source is heterogeneous across auditors, protocols, report formats and severity levels. This is a curated corpus built from public reports, not a pre-existing benchmark with official splits. See [DATASET_NOTICE.md](DATASET_NOTICE.md) for source-rights and redistribution notes.

Cleanup removes code blocks, URLs, images, tables and short lines; inline code is replaced with `identifier`. Text is lowercased and tokenized into English words and `. , ! ? ; :`. Numbers are removed. Exact duplicate cleaned sections and sections below 30 tokens are discarded. This intentionally sacrifices code identifiers and numeric precision to focus on prose.

| Split | Reports | Protocol groups | Tokens | Windows used | Unknown token share |
|---|---:|---:|---:|---:|---:|
| Train | 387 | 294 | 501,620 | 50,000 | 3.47% |
| Validation | 48 | 37 | 76,859 | 10,000 | 5.78% |
| Test | 48 | 37 | 51,873 | 10,000 | 5.79% |

Protocol groups are split before windows are created. Grouping uses the first protocol-name segment in dated filenames; it is a reproducible heuristic, not a manually verified protocol-family taxonomy. Exact duplicate section filtering does not eliminate all paraphrases or related findings. The training-only vocabulary has 4,000 tokens including `<unk>`. Each example has 24 context tokens and one next-token target; stride is three. No window crosses a finding boundary. Windows are uniformly subsampled with seed 42 to bound CPU cost; the full corpus supplies 136,015/20,660/14,026 candidate windows.

`results/source_manifest.json` records source URLs, original-file SHA-256 hashes, auditor, protocol group and split without reproducing report prose.

## Models and experimental controls

| Model | Recurrent mechanism | Context handling |
|---|---|---|
| Simple RNN | tanh hidden state | Left to right |
| Bidirectional RNN | Two tanh RNNs | Both directions within the known prefix only |
| LSTM | Gates and cell state | Left to right |
| GRU | Reset and update gates | Left to right |

All models use embeddings of width 64, one recurrent layer with 96 hidden units per direction, dropout 0.15 and a linear vocabulary classifier. Bidirectional RNN concatenates the final forward/backward hidden states; it never receives the next-word target as input. It remains an autoregressive generator because each output is sampled from only the known preceding text.

Controls: identical data, vocabulary, context, training shuffle seed, 12 epochs, Adam learning rate 0.003, batch size 256 and gradient clipping at norm 1.0. Each model's minimum validation-loss checkpoint is used for final test evaluation. Equal hidden width does not equal equal parameter counts; counts and timing are reported. The experiment uses one training seed (42) and three generation seeds (101, 202, 303), so it does not establish statistical significance across training runs.

## Run locally

Use Python 3.12, Git and a virtual environment. The recorded experiment used PyTorch 2.14.1+cpu and two CPU threads. No API keys, wallet, hosted database or paid service is needed.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m auditscribe.data --download
python -m unittest discover -s tests -v
python -m auditscribe.train
python -m auditscribe.analyze
python -m auditscribe.report
python -m auditscribe.generate --model gru --temperature 1.0 --prompt "the contract does not validate"
```

Run from the project root. Network access is required for the first dataset download and dependency installation. `AuditScribe.ipynb` is an unexecuted optional notebook wrapper around the same commands; install Jupyter separately to use it. It contains no invented execution outputs. Measured outputs are included in `results/`.

## Database

SQLite stores `documents`, `passages`, `runs` and `generations` locally at `data/processed/auditscribe.sqlite`. Source provenance, splits, metrics and generations are queryable without a database server. NumPy arrays are used for efficient model minibatches; SQLite is storage, not the neural-network training engine. Downloaded text, database, arrays and model checkpoints are excluded from the public package and regenerated by the commands above.

## Results and analysis

The measured run selected **LSTM** by validation loss. It also had the lowest test perplexity (156.72) and highest test accuracy (17.84%), compared with the unigram baseline at 357.41 perplexity and 7.18% accuracy. This is a single-seed comparison, not proof of universal superiority.

| Model | Test loss | Perplexity | Accuracy | Training seconds |
|---|---:|---:|---:|---:|
| RNN | 5.1033 | 164.56 | 16.98% | 51.8 |
| BIRNN | 5.2407 | 188.80 | 16.06% | 88.4 |
| LSTM | 5.0545 | 156.72 | 17.84% | 82.8 |
| GRU | 5.0819 | 161.07 | 17.47% | 91.1 |

At temperature 0.5, outputs are more formulaic and repetitive. At 1.0 they become more varied but often lose contextual consistency. At 1.5 every model averages a distinct-bigram ratio of 1.0, yet inspected text is fragmented and incoherent. Lexical novelty is therefore not equivalent to quality. LSTM is the preferred likelihood baseline for this corpus, while all four models remain unreliable as security report writers.

See [RESULTS.md](RESULTS.md) for the measured comparison, temperature analysis, qualitative review and conclusion. Files include:

- `results/metrics.csv`: final test metrics and model sizes/timings.
- `results/history.csv`: all 48 epoch records.
- `results/baseline.json`: smoothed frequency-only baseline.
- `results/temperature_summary.csv`: nine samples per model/temperature combination.
- `results/generated_samples.md`: every unedited generated continuation.
- `results/config.json`: settings and execution environment.

![Training and validation curves](results/learning_curves.png)
![Model comparison](results/model_comparison.png)
![Temperature analysis](results/temperature_analysis.png)

Cross entropy uses natural logarithms; perplexity is its exponential. Accuracy includes punctuation and unknown-token targets, while known-target accuracy excludes `<unk>` targets. Generated outputs suppress `<unk>` and use the full remaining vocabulary without top-k/top-p filtering. Distinct-1/2 and repeated trigram fraction quantify lexical diversity and repetition, not factual correctness. Human qualitative observations in RESULTS.md are explicitly separate from measured metrics.

## Repository map

- `auditscribe/data.py`: source acquisition, cleaning, grouped split, vocabulary, windows, SQLite.
- `auditscribe/model.py`: the four recurrent architectures.
- `auditscribe/train.py`: training, validation selection, evaluation, sampling and database recording.
- `auditscribe/analyze.py`: plots and temperature aggregation.
- `auditscribe/report.py`: rebuild the numerical report from run outputs.
- `auditscribe/generate.py`: prompt-based generation command.
- `tests/test_pipeline.py`: split and window invariants, cleanup and gradient checks.
- `WALKTHROUGH.md`: step-by-step explanation and viva preparation.
- `AuditScribe.ipynb`: optional interactive execution wrapper.

## Limitations and future work

Small recurrent models trained on a bounded sample produce grammatical fragments but may be repetitive, incoherent or technically false. The corpus reflects auditors' writing styles and selected public findings; it is not representative of all smart-contract security. Single-seed model training, heuristic protocol grouping, discarded identifiers/numbers and vocabulary truncation limit the strength of conclusions. An extended study should use reviewed protocol families, multiple training seeds, more data and a preregistered tuning budget. A future source-grounded retrieval assistant would be a separate project requiring factual validation.
