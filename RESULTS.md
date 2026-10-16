# AuditScribe results

## Experimental scope

This is one actual 12-epoch training run per architecture using training seed 42, the same 50,000 training examples and 10,000 examples in each evaluation split. The source revision and full settings are recorded in `results/dataset_summary.json` and `results/config.json`. No pretrained language model or external text-generation API was used. All values below are computed from the saved run outputs.

## Model comparison

| Model | Parameters | Selected epoch | Validation loss | Test loss | Test perplexity | Test accuracy | Known-target accuracy | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RNN | 659,552 | 4 | 5.0701 | 5.1033 | 164.56 | 16.98% | 17.44% | 51.8 |
| BIRNN | 1,059,104 | 3 | 5.2072 | 5.2407 | 188.80 | 16.06% | 16.46% | 88.4 |
| LSTM | 706,208 | 7 | 5.0297 | 5.0545 | 156.72 | 17.84% | 18.27% | 82.8 |
| GRU | 690,656 | 4 | 5.0361 | 5.0819 | 161.07 | 17.47% | 17.38% | 91.1 |

The training-target smoothed unigram baseline has test loss 5.8789, perplexity 357.41 and accuracy 7.18%. It assigns probabilities using training-target frequencies with add-one smoothing and never uses the context.

**Validation-selected model: LSTM.** Its checkpoint achieved the lowest validation loss. The lowest held-out test perplexity belongs to LSTM; this is reported as a final comparison, not used to tune the training setup.

Training time sums minibatch training time over all 12 epochs, including forward/backward passes and optimizer updates. It excludes validation and text sampling. `fit_wall_seconds` in the machine-readable results also includes validation and checkpoint writes. This shared CPU environment is not a dedicated timing benchmark.

## Learning behaviour

The saved learning curves show falling training loss and a validation minimum before epoch 12 for all four architectures. That divergence is consistent with overfitting. Selecting validation checkpoints avoids reporting the last, more overfit checkpoint as the default. The Bidirectional RNN has more parameters because its two context encoders feed a classifier with twice the input width. Comparable optimization settings do not imply equal capacity or equal tuning effort.

![Learning curves](results/learning_curves.png)

## Temperature experiment

Each model uses the same three prompts, temperatures 0.5/1.0/1.5, generation seeds 101/202/303 and output length of 60 tokens. This gives 108 unedited continuations. `<unk>` is suppressed only during generation. Sampling otherwise uses the entire vocabulary without top-k or top-p filtering. Short seed prompts are encoded at their actual length; the rolling context grows to 24 tokens.

| Model | Temperature | Samples | Mean distinct unigram | Mean distinct bigram | Mean repeated trigram |
|---|---:|---:|---:|---:|---:|
| RNN | 0.5 | 9 | 0.467 | 0.795 | 0.038 |
| RNN | 1.0 | 9 | 0.750 | 0.994 | 0.000 |
| RNN | 1.5 | 9 | 0.900 | 1.000 | 0.000 |
| BIRNN | 0.5 | 9 | 0.411 | 0.702 | 0.142 |
| BIRNN | 1.0 | 9 | 0.776 | 0.989 | 0.000 |
| BIRNN | 1.5 | 9 | 0.915 | 1.000 | 0.000 |
| LSTM | 0.5 | 9 | 0.441 | 0.718 | 0.140 |
| LSTM | 1.0 | 9 | 0.793 | 0.983 | 0.000 |
| LSTM | 1.5 | 9 | 0.902 | 1.000 | 0.000 |
| GRU | 0.5 | 9 | 0.493 | 0.800 | 0.061 |
| GRU | 1.0 | 9 | 0.776 | 0.994 | 0.000 |
| GRU | 1.5 | 9 | 0.920 | 1.000 | 0.000 |

Distinct-n is the number of unique generated n-grams divided by the number of generated n-grams, calculated per sample and then averaged. Repeated trigram fraction is one minus distinct-3. Prompts are excluded from these lexical metrics. Higher diversity can simply reflect random word combinations, so it must be considered together with the qualitative review.

![Temperature analysis](results/temperature_analysis.png)

## Qualitative assessment

The following observations are an exploratory manual inspection, not blinded human ratings. They focus on the common prompt `the contract does not validate` with generation seed 101 across temperatures and models; all other outputs remain available for review in `results/generated_samples.md`.

- At temperature 0.5, the RNN and Bidirectional RNN often form recognizable local clauses about an owner or function, but reuse generic templates. The LSTM continuation repeatedly uses `identifier`, including consecutive repetitions. Local grammar does not translate into a coherent, actionable finding.
- At temperature 1.0, outputs introduce a broader range of audit vocabulary and occasional recommendation phrasing, but drift between unrelated topics and contain grammatical errors. The LSTM sample moves between a recipient, yield, a request and a constructor without a consistent causal explanation.
- At temperature 1.5, the inspected RNN, Bidirectional RNN and LSTM samples include rare vocabulary and fragmented word sequences. Their lexical diversity should not be mistaken for improved contextual consistency or useful security reasoning.
- The GRU sample at 0.5 also repeats generic identifier phrases; at 1.0 it shifts between an audit, a recipient, a proposal, yield and migration. At 1.5 it becomes similarly fragmented. Its mean low-temperature distinct-bigram ratio is slightly higher than the other models, but this does not establish better semantic quality.
- Contextual consistency is weak: the initial validation prompt rarely develops into a complete explanation of what is validated, what fails and why it matters. No generated claim has been checked against a real contract.

For readable demonstration fragments, 0.5 is a conservative starting point, with repetition clearly visible. Temperature 1.0 is useful for demonstrating the diversity/coherence trade-off; 1.5 is useful chiefly to expose failure modes. This is a qualitative recommendation, not an objectively optimized temperature.

## Conclusion

For this fixed-budget experiment, **LSTM is the recommended model**, selected by validation loss and assessed alongside the held-out comparison. It is an experimental domain-language baseline, not a production audit writer. The generated text does not justify claims of factual security expertise. A stronger claim about architecture superiority would require multiple independent training seeds, confidence intervals, reviewed protocol-family splits and an equal tuning budget.

The project meets the required implementation and comparison scope: four recurrent architectures, reproducible text preprocessing and next-token sequences, quantitative loss/perplexity/accuracy/timing results, qualitative generation analysis, identical seed prompts at three temperatures, and an evidence-based model choice.
