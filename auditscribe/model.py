import torch
from torch import nn

class LanguageModel(nn.Module):
    def __init__(self, kind, vocab_size, embedding=64, hidden=96):
        super().__init__()
        if kind not in ['rnn','birnn','lstm','gru']: raise ValueError(kind)
        self.kind=kind
        self.embedding=nn.Embedding(vocab_size,embedding)
        cls={'rnn':nn.RNN,'birnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU}[kind]
        self.recurrent=cls(embedding,hidden,batch_first=True,bidirectional=kind=='birnn')
        self.dropout=nn.Dropout(.15)
        self.output=nn.Linear(hidden*(2 if kind=='birnn' else 1),vocab_size)
    def forward(self,x):
        _,state=self.recurrent(self.embedding(x))
        h=state[0] if isinstance(state,tuple) else state
        # Both directions encode ONLY the supplied past window, never target tokens.
        features=torch.cat((h[-2],h[-1]),dim=-1) if self.kind=='birnn' else h[-1]
        return self.output(self.dropout(features))
