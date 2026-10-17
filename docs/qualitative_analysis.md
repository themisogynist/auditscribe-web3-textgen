# Qualitative analysis of generated audit text

The comparison below inspects the fixed prompt `the contract does not validate`, generation seed 101 and all three temperatures for every model. These are descriptive observations, not blinded human ratings. Complete continuations for this comparison and all other prompts/seeds are retained in `results/generated_samples.txt` and `results/generated_samples.md`.

## Coherence and contextual consistency

| Model | Temperature 0.5 | Temperature 1.0 | Temperature 1.5 |
|---|---|---|---|
| Simple RNN | Starts with a relevant owner-check statement, then repeats identifier/function phrases and ends unfinished. | Moves from a recipient to yield and rewards; local security vocabulary lacks a consistent explanation. | Rare terms and broken syntax dominate; the original validation context is lost. |
| Bidirectional RNN | Includes recommendation phrasing, but the owner, loop, balance and constructor are not connected logically. | Shifts between recipients, borrowing and maintenance with grammatical errors. | Fragmented mentions of transfers, prices and other concepts have weak contextual consistency. |
| LSTM | Produces repeated grammatical fragments around `identifier function can be used`; repetition overwhelms meaning. | Contains recognizable audit phrases but drifts between yield, reward information and validation. | Greater word variety comes with incoherent combinations and unstable sentence structure. |
| GRU | Retains owner/address/contract vocabulary, but repeats generic statements and the phrase `as low as low`. | Mentions governance, signatures and migration without explaining a single vulnerability. | Broad vocabulary and low repetition do not produce a coherent causal account. |

At temperature 0.5, the Simple RNN continuation begins `the owner is not checked in the contract`. This is locally compatible with the prompt, but the remainder does not sustain a clear finding. The GRU continuation stays broadly within the access-control domain while repeating generic contract/address language. LSTM is the most repetitive of the four at this temperature according to the mean repeated-trigram measurement (0.167).

## Diversity and repetition

Higher temperatures flatten the token distribution and generally increase lexical diversity. At 0.5, mean distinct-bigram ratios range from 0.678 to 0.812 and repeated-trigram fractions from 0.075 to 0.167. At 1.0, distinct-bigram ratios rise to 0.983-0.996; repeated-trigram fractions are zero except for LSTM (0.0038). At 1.5, mean distinct-bigram ratios are at least 0.996 and repeated-trigram fractions are zero for every model.

These statistics measure repetition within individual 60-token continuations. A zero repeated-trigram fraction does not establish fluent or accurate prose. The examples at 1.5 demonstrate this distinction: they contain diverse vocabulary but frequently lose grammar and topic continuity.

## Temperature choice and model selection

Temperature 0.5 is the most defensible setting among those tested for short, recognizable audit-style fragments, although repetition remains substantial. Temperature 1.0 exposes the trade-off between variety and consistency. Temperature 1.5 is unsuitable for readable audit text in this experiment. No temperature yields reliably correct technical findings.

GRU is selected overall because it has the lowest validation loss, lowest test loss/perplexity and highest primary test accuracy. It also has the lowest mean repeated-trigram fraction at temperature 0.5 and ties Simple RNN for the highest distinct-bigram ratio at that temperature. It uses fewer parameters and less measured training time than LSTM in this run. The numerical gap between GRU and LSTM is small, and qualitative inspection does not establish decisive semantic superiority. The conclusion is limited to this corpus, configuration and training seed.

## Limitations

All four models can generate misleading or technically false statements. The experiment evaluates language modelling, not vulnerability detection or remediation correctness. Code identifiers and numbers were simplified during preprocessing; this limits technical specificity. Short seed contexts, a bounded vocabulary and one training seed also restrict conclusions. The protocol grouping is heuristic and does not eliminate all possible semantic overlap between related audits.
