# AuditScribe step by step

## 1 Understand the objective

Given the preceding 24 tokens from an audit finding, predict the next token. Feed each sampled prediction back into the context to continue writing. The domain is smart-contract security; the system does not inspect Solidity contracts or establish whether a vulnerability exists.

## 2 Prepare the data

Run `python -m auditscribe.data --download`. This downloads the pinned Solodit repository revision, selects reports with a Findings heading, removes fenced and indented code, strips links and tables, replaces inline code with `identifier`, lowercases text and retains English words plus six punctuation tokens. Findings shorter than 30 tokens and exact duplicate cleaned sections are removed. The token `identifier` deliberately loses function-name detail to reduce vocabulary sparsity.

The first segment of each dated report's protocol name defines a coarse group. Group assignment is seeded 80/10/10; report counts therefore need not be exactly 80/10/10. This heuristic reduces re-audit leakage but cannot prove unrelated names or forks are independent. The vocabulary is fitted only to training reports, capped at 4,000 entries with frequency at least three; unseen words map to `<unk>`.

A window holds 24 input tokens and one target. Windows advance by three positions and never cross finding boundaries. A deterministic subset limits CPU cost to 50,000 training windows and 10,000 per evaluation split. The processed SQLite database stores documents and passages; NumPy arrays feed the training loop.

## 3 Understand the four models

- Simple RNN uses a recurrent hidden state and tanh activation.
- Bidirectional RNN encodes the given past window in both directions and concatenates the two final states. The target remains outside that window, so there is no look-ahead to the token being predicted.
- LSTM uses gates and a cell state to regulate retained information.
- GRU uses update and reset gates with one recurrent state.

Each has 64-dimensional embeddings, 96 hidden units per direction, a dropout layer and a linear vocabulary classifier. Same hidden width does not imply equal parameter counts; counts are reported.

## 4 Train and select checkpoints

Run `python -m auditscribe.train`. All models see the same data, batch ordering seed, 12 epochs, Adam at 0.003, batch size 256 and gradient clipping at 1.0. Validation cross entropy selects one checkpoint per model. The test split is evaluated after that model's checkpoint selection. Training time excludes validation and generation; fit wall time includes validation and checkpoint writing. CPU timing varies with hardware and concurrent workloads.

## 5 Interpret metrics

Cross entropy is mean negative log probability of the correct next token. Perplexity is exp(cross entropy); lower is better with the same tokenizer and test set. Accuracy measures exact top-one next-token predictions, including punctuation and `<unk>`. Known-target accuracy excludes `<unk>` targets from its denominator. A smoothed unigram baseline checks whether context-sensitive models improve on a frequency-only predictor.

## 6 Compare temperatures

Sampling uses softmax(logits / temperature). At 0.5 probability mass is sharper; at 1.0 it is unchanged; at 1.5 it is flatter. These are tendencies, not guarantees about coherence. There is no top-k or top-p filter. Three prompts and three sampling seeds produce nine outputs per temperature per model. Distinct bigram ratio measures lexical variety; repeated trigram fraction measures repetition. Neither proves semantic correctness.

## 7 Review and explain the result

Run `python -m auditscribe.analyze`, then read `RESULTS.md` and `results/generated_samples.md`. Explain actual strengths and failures, including grammatical mistakes, invented relationships and repetition. A single training seed is an exploratory comparison, not statistical proof that one architecture is universally best.

## 8 Publish and submit

Create a GitHub repository named `auditscribe-web3-textgen`, then upload the package contents. Keep downloaded report text, local database, cached arrays and checkpoints out of the public repository. Include source links and the downloader, measured outputs, code, README and analysis. Verify the repository is public or grant the evaluator access. Fill the verified repository URL into the submission document, export it to PDF and upload that PDF to LMS.

## Viva questions

Why group before windowing? Random window splits can put almost identical overlapping context into both train and test.

Why SQLite? Documents, source provenance, experiments and generations have simple relational structure; local storage requires no account or service.

Why can a bidirectional model generate autoregressively? It is bidirectional only inside the known prefix. Each next token is still sampled sequentially from that prefix.

Why not claim an AI auditor? The objective is next-word prediction on audit prose, not code analysis or verified vulnerability detection.

Why might lower perplexity still yield poor text? Teacher-forced evaluation conditions on real prefixes; generation conditions on its own accumulating mistakes.
