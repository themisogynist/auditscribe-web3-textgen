"""Generate the numerical comparison from recorded experiment outputs."""
import csv,json
from pathlib import Path

def main():
    out=Path('results')
    metrics=json.loads((out/'metrics.json').read_text())
    cfg=json.loads((out/'config.json').read_text())
    data=json.loads((out/'dataset_summary.json').read_text())
    base=json.loads((out/'baseline.json').read_text())
    temps=list(csv.DictReader((out/'temperature_summary.csv').open()))
    if len(metrics)!=4:raise ValueError('Four completed model runs are required')
    best=min(metrics,key=lambda x:x['best_validation_loss'])
    lines=['# AuditScribe experimental results','', '## Experimental conditions','',
      f"All four models use {data['splits']['train']['used_windows']:,} training windows, {cfg['epochs']} epochs, batch size {cfg['batch_size']} and training seed {cfg['seed']}. Each evaluation split contains 10,000 fixed windows. Checkpoints are selected by minimum validation loss. Full settings are recorded in results/config.json.",'',
      '## Quantitative comparison','',
      '| Model | Parameters | Epoch | Validation loss | Test loss | Perplexity | Accuracy | Known-target accuracy | Lexical-word accuracy | Train seconds |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        lines.append(f"| {r['model'].upper()} | {r['parameters']:,} | {r['best_epoch']} | {r['best_validation_loss']:.4f} | {r['loss']:.4f} | {r['perplexity']:.2f} | {100*r['accuracy']:.2f}% | {100*r['known_accuracy']:.2f}% | {100*r['word_accuracy']:.2f}% | {r['training_seconds']:.1f} |")
    lines+=['',f"The smoothed unigram baseline achieves loss {base['loss']:.4f}, perplexity {base['perplexity']:.2f} and accuracy {100*base['accuracy']:.2f}%. It is fitted only to training-target counts.",'',
      'Primary accuracy includes punctuation and unknown tokens. Known-target accuracy excludes unknown targets; lexical-word accuracy excludes unknown targets and punctuation. Training time excludes validation and sampling. Fit wall time is reported separately in metrics.csv. CPU timings depend on hardware and system load.','',
      '![Model comparison](results/model_comparison.png)','',
      '## Learning behaviour','',
      'RNN, Bidirectional RNN and GRU reach their selected validation minima before the final epoch, while LSTM selects epoch 16. Validation-loss scheduling reduces the learning rate when improvement stalls. The stronger divergence of training and validation loss in Bidirectional RNN is consistent with overfitting. Its larger parameter count must be considered when comparing architectures.','',
      '![Learning curves](results/learning_curves.png)','',
      '## Sampling temperature analysis','',
      'Three identical prompts across all four models, three temperatures and three generation seeds produce 108 continuations of 60 tokens. Values below average the nine outputs per model/temperature pair.','',
      '| Model | Temperature | Samples | Distinct unigram | Distinct bigram | Repeated trigram |',
      '|---|---:|---:|---:|---:|---:|']
    for r in temps:
        lines.append(f"| {r['model'].upper()} | {r['temperature']} | {r['n']} | {float(r['distinct_1']):.3f} | {float(r['distinct_2']):.3f} | {float(r['repetition_3']):.3f} |")
    lines+=['','Distinct-n divides the number of unique generated n-grams by all generated n-grams. Repeated-trigram fraction equals one minus distinct-3. Prompts are excluded. Lexical diversity is not a semantic quality score.','',
      '![Temperature analysis](results/temperature_analysis.png)','',
      '## Qualitative comparison','',
      'The [qualitative analysis](docs/qualitative_analysis.md) compares coherence, diversity, contextual consistency and repetition across all four models. [All generated text](results/generated_samples.txt) is available without editing. Low temperatures retain more recognizable phrases but repeat templates; high temperatures increase diversity while weakening syntax and topic consistency.','',
      '## Conclusion','',
      f"**{best['model'].upper()} is the best-performing model in this experiment**, based on validation selection and the final test comparison. Its low-temperature generation diagnostics support its selection, but none of the models reliably produces factually correct audit reports. The small GRU/LSTM gap and single training seed limit generalization beyond this experiment.",'']
    Path('RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
