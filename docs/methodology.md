# Methodology

## Dataset characteristics and preprocessing

The corpus is constructed from Markdown audit reports in the Solodit Content Repository at the revision specified in `auditscribe/data.py`. It contains natural-language descriptions of smart-contract vulnerabilities, impacts and mitigations across heterogeneous protocols, auditors and severity levels. Reports without a recognized Findings section are excluded. The resulting corpus contains 483 reports and 5,140 cleaned sections.

Fenced and indented code, images, URLs, HTML tags and Markdown table rows are removed. Link labels remain as text. Inline code is replaced by `identifier` to reduce sparsity from contract-specific names. Lines below five tokens and sections below 30 tokens are discarded. Text is lowercased, numeric strings are removed, and the tokenizer retains English words (including internal apostrophes) and six punctuation marks. Exact duplicate cleaned sections are removed globally before splitting; no model statistics are fitted during this deduplication.

## Protocol grouping and vocabulary

The first segment of the protocol name in each dated filename defines a coarse group. Groups are randomly assigned to train, validation and test in an 80/10/10 ratio using seed 42. This yields 387/48/48 reports. All retained sections from one group remain in one split. This reduces leakage from repeated audits, but aliases and related protocol families may still cross groups because the grouping is heuristic.

Only training text is used to fit the vocabulary. It contains up to 4,000 entries with minimum frequency three, including one unknown token. Validation and test vocabulary coverage are reported separately. Unknown words map to `<unk>`.

## Input and target construction

For a cleaned sequence of tokens w1, w2, ..., wn, a training example uses the preceding 24 tokens as input and the immediately following token as target. Windows advance by three positions and never cross section boundaries. All 136,015 training windows are used. Validation and test are each deterministically sampled to 10,000 windows from their respective candidate windows.

This is word-level language modelling with punctuation represented as separate tokens. The primary prediction accuracy includes punctuation and unknown-token targets; additional known-target and lexical-word accuracies clarify their contribution. Vocabulary, validation examples and test examples are unchanged from the preliminary experiment.

## Four recurrent architectures

Each architecture uses a 64-dimensional embedding, one recurrent layer with 96 hidden units per direction, dropout 0.25 and a linear classifier over the vocabulary. Simple RNN uses a tanh recurrence. LSTM uses input, forget and output gates with a cell state. GRU uses update and reset gates. Bidirectional RNN concatenates the final states of two tanh recurrences over the same known prefix.

The bidirectional encoder sees only the preceding context window. The next-word target is never part of that input. It is therefore valid to use this encoder for sequential autoregressive generation, despite its bidirectional processing within the known context.

## Comparable training conditions

All four architectures use the same training windows, vocabulary, batch ordering seed, 16 epochs, batch size 512, AdamW optimizer, initial learning rate 0.0015, weight decay 0.0001 and gradient norm clipping at 1.0. The same ReduceLROnPlateau rule halves learning rate after two non-improving validation epochs, with a floor of 0.0001. Model-specific validation behaviour can produce different learning-rate trajectories; these are recorded in the history files.

The lowest validation cross entropy selects the checkpoint for each model. The final test evaluation uses that checkpoint. Parameter counts differ, especially for Bidirectional RNN; the experiment controls hidden width and optimization rules rather than exact parameter budget. One training seed is used, so differences are descriptive rather than statistically established.

## Evaluation

Cross entropy is the mean negative natural logarithm of the target probability. Perplexity equals exp(cross entropy). Top-one accuracy is the fraction of targets matching the highest-probability token. Known-target accuracy excludes unknown targets. Lexical-word accuracy excludes unknown targets and punctuation. The additional denominators do not alter the primary accuracy or checkpoint selection.

Training time sums forward/backward passes and optimization over all 16 epochs. Fit wall time also includes validation and checkpoint writes. Text generation time is excluded. A smoothed unigram baseline is fitted to training-target counts with add-one smoothing.

## Autoregressive generation and temperature

The model encodes a prompt, divides its next-token logits by temperature, samples from the resulting softmax distribution, appends the sampled token and repeats. Context is truncated to the latest 24 tokens. Short prompts initially use their available context. Unknown-token output is suppressed during sampling, but not during evaluation. No top-k or top-p truncation is applied.

The same three prompts, three temperatures (0.5, 1.0 and 1.5), three generation seeds (101, 202 and 303), and 60-token continuation length are used for every model. This produces 108 outputs. All outputs are retained without editing. Distinct unigram/bigram ratios and repeated trigram fraction are computed on generated tokens only, then averaged over the nine samples per model/temperature pair.

## Storage and reproducibility

SQLite stores document provenance, cleaned passages, run configuration, metrics and generated outputs. NumPy arrays supply minibatches. Source files, the local database, arrays and checkpoints are excluded from Git; the acquisition and training commands regenerate them. The source manifest records each report's URL, hash, auditor, protocol group and split. Public source access does not imply unrestricted redistribution; dataset terms are described in DATASET_NOTICE.md.
