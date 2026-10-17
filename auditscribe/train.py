"""Comparable CPU/GPU experiment. Validation chooses checkpoints; test is final-only."""
import argparse, collections, csv, json, math, platform, random, sqlite3, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .model import LanguageModel
from .data import tokenize

def evaluate(model,loader,device,word_ids=None):
    model.eval(); total=correct=known=known_correct=word_total=word_correct=0; loss=0.
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device),y.to(device); logits=model(x); pred=logits.argmax(1)
            loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item();total+=len(y);correct+=(pred==y).sum().item()
            mask=y!=0; known+=mask.sum().item();known_correct+=((pred==y)&mask).sum().item()
            if word_ids is not None:
                wm=word_ids[y];word_total+=wm.sum().item();word_correct+=((pred==y)&wm).sum().item()
    result=dict(loss=loss/total,perplexity=math.exp(loss/total),accuracy=correct/total,known_accuracy=known_correct/max(known,1))
    if word_ids is not None:result['word_accuracy']=word_correct/max(word_total,1)
    return result

def sample(model,vocab,prompt,temperature,seed,device,context,words=60):
    rng=torch.Generator(device=device).manual_seed(seed); lookup={w:i for i,w in enumerate(vocab)}
    ids=[lookup.get(w,0) for w in tokenize(prompt)]; generated=[];model.eval()
    with torch.inference_mode():
        for _ in range(words):
            logits=model(torch.tensor([ids[-context:]],device=device))[0]/temperature
            logits[0]=-float('inf') # Suppress unknown only at generation, never in evaluation.
            nxt=torch.multinomial(torch.softmax(logits,0),1,generator=rng).item()
            ids.append(nxt); generated.append(vocab[nxt])
    pairs=list(zip(generated,generated[1:]));triples=list(zip(generated,generated[1:],generated[2:]))
    return dict(text=' '.join(generated),distinct_1=len(set(generated))/len(generated),distinct_2=len(set(pairs))/len(pairs),repetition_3=1-len(set(triples))/len(triples))

def run(data,out,epochs=16,seed=42,threads=2):
    torch.set_num_threads(threads);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.use_deterministic_algorithms(True)
    data,out=Path(data),Path(out);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    vocab=json.loads((data/'vocab.json').read_text());arrays=np.load(data/'windows.npz');context=arrays['train'].shape[1]-1
    word_ids=torch.tensor([w not in ['<unk>','.',',','!','?',';',':'] for w in vocab],device=device)
    datasets={k:TensorDataset(torch.from_numpy(arrays[k][:,:-1].copy()),torch.from_numpy(arrays[k][:,-1].copy())) for k in arrays.files}
    valid=DataLoader(datasets['validation'],batch_size=512);test=DataLoader(datasets['test'],batch_size=512)
    config=dict(epochs=epochs,seed=seed,batch_size=512,embedding=64,hidden=96,dropout=.25,optimizer='AdamW',learning_rate=.0015,weight_decay=.0001,scheduler='ReduceLROnPlateau factor 0.5 patience 2 min_lr 0.0001',gradient_clip=1.0,context=context,torch=torch.__version__,python=platform.python_version(),device=device,threads=threads,checkpoint='minimum validation cross entropy',temperature=[.5,1.,1.5],generation_length=60)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    results=[];history=[];samples=[]
    for kind in ['rnn','birnn','lstm','gru']:
        torch.manual_seed(seed)
        model=LanguageModel(kind,len(vocab)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,min_lr=.0001)
        loader=DataLoader(datasets['train'],batch_size=512,shuffle=True,generator=torch.Generator().manual_seed(seed))
        best=float('inf');best_epoch=0;start=time.perf_counter();training_seconds=0
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0.;n=0;t=time.perf_counter()
            for x,y in loader:
                x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);logits=model(x)
                loss=nn.functional.cross_entropy(logits,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                loss_sum+=loss.item()*len(y);n+=len(y)
            if device=='cuda':torch.cuda.synchronize()
            epoch_seconds=time.perf_counter()-t;training_seconds+=epoch_seconds
            val=evaluate(model,valid,device)
            row=dict(model=kind,epoch=epoch,train_loss=loss_sum/n,validation_loss=val['loss'],validation_accuracy=val['accuracy'],training_seconds=epoch_seconds,learning_rate=opt.param_groups[0]['lr'])
            scheduler.step(val['loss'])
            history.append(row);print(json.dumps(row),flush=True)
            if val['loss']<best:
                best=val['loss'];best_epoch=epoch;torch.save(model.state_dict(),out/'checkpoints'/f'{kind}.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2))
        elapsed=time.perf_counter()-start
        model.load_state_dict(torch.load(out/'checkpoints'/f'{kind}.pt',map_location=device,weights_only=True))
        metrics=evaluate(model,test,device,word_ids)
        results.append(dict(model=kind,parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,best_validation_loss=best,training_seconds=training_seconds,fit_wall_seconds=elapsed,**metrics))
        prompts=['the contract does not validate','an attacker can manipulate the','to mitigate this vulnerability']
        for prompt in prompts:
            for temp in [.5,1.,1.5]:
                for generation_seed in [101,202,303]:
                    samples.append(dict(model=kind,prompt=prompt,temperature=temp,seed=generation_seed,**sample(model,vocab,prompt,temp,generation_seed,device,context)))
        (out/'metrics.json').write_text(json.dumps(results,indent=2));(out/'samples.json').write_text(json.dumps(samples,indent=2))
    # Smoothed unigram distribution fitted only on the exact training targets.
    freq=np.bincount(arrays['train'][:,-1],minlength=len(vocab))+1
    probs=freq/freq.sum();targets=arrays['test'][:,-1];base_loss=float(-np.log(probs[targets]).mean())
    (out/'baseline.json').write_text(json.dumps(dict(model='smoothed unigram',loss=base_loss,perplexity=math.exp(base_loss),accuracy=float((targets==probs.argmax()).mean())),indent=2))
    for name,rows in [('metrics',results),('history',history),('samples',samples)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    db=sqlite3.connect(data/'auditscribe.sqlite')
    db.executescript('DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS generations; CREATE TABLE runs(model TEXT PRIMARY KEY,config_json TEXT,metrics_json TEXT); CREATE TABLE generations(model TEXT,prompt TEXT,temperature REAL,seed INTEGER,text TEXT,distinct_1 REAL,distinct_2 REAL,repetition_3 REAL);')
    db.executemany('INSERT INTO runs VALUES (?,?,?)',[(r['model'],json.dumps(config),json.dumps(r)) for r in results])
    db.executemany('INSERT INTO generations VALUES (?,?,?,?,?,?,?,?)',[tuple(s[k] for k in ['model','prompt','temperature','seed','text','distinct_1','distinct_2','repetition_3']) for s in samples]);db.commit();db.close()
    print('Experiment complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed');p.add_argument('--out',default='results');p.add_argument('--epochs',type=int,default=16);p.add_argument('--threads',type=int,default=2);a=p.parse_args();run(a.data,a.out,a.epochs,threads=a.threads)
