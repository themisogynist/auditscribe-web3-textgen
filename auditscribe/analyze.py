"""Create plots and aggregate generation diagnostics from measured run outputs."""
import collections,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=Path('results');metrics=json.loads((out/'metrics.json').read_text());history=json.loads((out/'history.json').read_text());samples=json.loads((out/'samples.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'rnn':'#58718a','birnn':'#a45b78','lstm':'#148779','gru':'#b27925'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for kind in colors:
        rows=[r for r in history if r['model']==kind]
        axes[0].plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label=kind.upper(),color=colors[kind])
        axes[1].plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],label=kind.upper(),color=colors[kind])
    for ax,title in zip(axes,['Training cross entropy','Validation cross entropy']):
        ax.set(xlabel='Epoch',ylabel='Loss (nats)',title=title);ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,['perplexity','accuracy','training_seconds'],['Test perplexity (lower is better)','Test accuracy','Training time (seconds)']):
        ax.bar([r['model'].upper() for r in metrics],[r[key] for r in metrics],color=[colors[r['model']] for r in metrics]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'model_comparison.png',dpi=180);plt.close(fig)
    agg=[]
    for kind in colors:
        for t in [.5,1.,1.5]:
            rows=[r for r in samples if r['model']==kind and r['temperature']==t]
            agg.append(dict(model=kind,temperature=t,n=len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in ['distinct_1','distinct_2','repetition_3']}))
    with (out/'temperature_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for kind in colors:
        rows=[r for r in agg if r['model']==kind]
        for ax,key in zip(axes,['distinct_2','repetition_3']):ax.plot([r['temperature'] for r in rows],[r[key] for r in rows],marker='o',label=kind.upper(),color=colors[kind])
    for ax,title in zip(axes,['Mean distinct bigram ratio','Mean repeated trigram fraction']):ax.set(xlabel='Temperature',ylabel='Fraction',title=title,xticks=[.5,1.,1.5]);ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'temperature_analysis.png',dpi=180);plt.close(fig)
    text=['# Generated continuations','', 'All 108 outputs are unedited. Each continuation has 60 generated tokens; prompts are shown separately. Unknown tokens are suppressed only when sampling.','']
    for s in samples:
        text += [f"## {s['model'].upper()} | temperature {s['temperature']} | seed {s['seed']}",f"Prompt: {s['prompt']}",'',s['text'],'']
    (out/'generated_samples.md').write_text('\n'.join(text))
    (out/'generated_samples.txt').write_text('\n'.join(line.lstrip('# ') if line.startswith('#') else line for line in text))
    print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
