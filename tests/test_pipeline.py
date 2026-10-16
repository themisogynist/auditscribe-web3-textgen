import json, unittest
from pathlib import Path
import numpy as np
import torch
from auditscribe.data import clean
from auditscribe.model import LanguageModel

class PipelineTests(unittest.TestCase):
    def test_cleanup_excludes_fenced_code(self):
        text='The attacker can withdraw all deposited tokens.\n```solidity\nsecretcode xyz\n```\nThe protocol must validate the caller before withdrawal.'
        self.assertNotIn('secretcode',clean(text))
        self.assertIn('attacker',clean(text))
    def test_group_disjointness_and_windows(self):
        path=Path('data/processed')
        if not path.exists():self.skipTest('Prepare the dataset first')
        rows=json.loads((path/'manifest.json').read_text());groups={}
        for row in rows:
            groups.setdefault(row['protocol'],set()).add(row['split'])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        vocab=json.loads((path/'vocab.json').read_text())
        for arr in np.load(path/'windows.npz').values():
            self.assertEqual(arr.shape[1],25)
            self.assertGreaterEqual(arr.min(),0);self.assertLess(arr.max(),len(vocab))
    def test_all_models_train_next_token(self):
        x=torch.randint(0,20,(3,24));y=torch.tensor([1,2,3])
        for kind in ['rnn','birnn','lstm','gru']:
            m=LanguageModel(kind,20);out=m(x)
            self.assertEqual(out.shape,(3,20));torch.nn.functional.cross_entropy(out,y).backward()
            self.assertIsNotNone(m.embedding.weight.grad)

if __name__=='__main__':unittest.main()
