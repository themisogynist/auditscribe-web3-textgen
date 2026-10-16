"""Generate a continuation from a locally trained checkpoint."""
import argparse,json
from pathlib import Path
import torch
from .model import LanguageModel
from .train import sample

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['rnn','birnn','lstm','gru'],default='gru');p.add_argument('--prompt',default='the contract does not validate');p.add_argument('--temperature',type=float,default=1.);p.add_argument('--seed',type=int,default=101);a=p.parse_args()
    if a.temperature<=0:p.error('Temperature must be positive')
    torch.set_num_threads(2)
    vocab=json.loads(Path('data/processed/vocab.json').read_text());cfg=json.loads(Path('results/config.json').read_text())
    model=LanguageModel(a.model,len(vocab));model.load_state_dict(torch.load(f'results/checkpoints/{a.model}.pt',map_location='cpu',weights_only=True))
    result=sample(model,vocab,a.prompt,a.temperature,a.seed,'cpu',cfg['context'])
    print(a.prompt+' '+result['text'])
    print('\nExperimental continuation; factual and technical accuracy are not guaranteed.')
if __name__=='__main__':main()
