"""Pinned audit corpus, prose cleanup, grouped splitting and SQLite persistence."""
import argparse, collections, hashlib, json, random, re, sqlite3, subprocess
from pathlib import Path
from urllib.parse import quote

SOURCE = 'https://github.com/solodit/solodit_content'
REVISION = '4d5330d12978c7a8fc2aab5b171112784fb671bd'
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")

def tokenize(text):
    return TOKEN.findall(text.lower())

def clean(text):
    # Reports may contain code and links; the task models prose, not Solidity.
    text = re.sub(r'```.*?```|~~~.*?~~~', ' ', text, flags=re.S)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'`[^`]+`', ' identifier ', text)
    lines = []
    for line in text.splitlines():
        if line.startswith(('    ', '\t')) or '|' in line:
            continue
        line = re.sub(r'^\s*[#>*\-\d.]+\s*', '', line)
        if len(tokenize(line)) >= 5:
            lines.append(line)
    return ' '.join(tokenize(' '.join(lines)))

def protocol_key(path):
    # Coarse first protocol name groups dated/versioned re-audits together.
    stem = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', path.stem.lower())
    return re.split(r'[-_\s]', stem)[0]

def acquire(dest):
    dest = Path(dest)
    if not (dest / '.git').exists():
        subprocess.run(['git', 'clone', SOURCE + '.git', str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', REVISION], check=True)
    return dest

def prepare(source, out, seed=42, context=24, max_vocab=4000, max_train=50000):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise ValueError(f'Expected pinned revision {REVISION}; got {actual}')
    docs, seen = [], set()
    for p in sorted((Path(source) / 'reports').rglob('*.md')):
        if p.name.lower() == 'readme.md': continue
        raw = p.read_text(errors='replace')
        parts = re.split(r'^#\s+Findings\s*$', raw, maxsplit=1, flags=re.M|re.I)
        if len(parts) != 2: continue
        # One paragraph per finding subsection; removes exact repeated blocks globally.
        blocks=[]
        for section in re.split(r'^###\s+', parts[1], flags=re.M):
            prose=clean(section)
            digest=hashlib.sha256(prose.encode()).hexdigest()
            if len(prose.split()) < 30 or digest in seen: continue
            seen.add(digest); blocks.append(prose)
        if not blocks: continue
        rel=p.relative_to(source).as_posix()
        docs.append(dict(path=rel, protocol=protocol_key(p), auditor=p.parent.name,
                         url=SOURCE+'/blob/'+REVISION+'/'+quote(rel),
                         sha256=hashlib.sha256(raw.encode()).hexdigest(), blocks=blocks))
    groups=sorted({d['protocol'] for d in docs}); random.Random(seed).shuffle(groups)
    n=len(groups); mapping={g:('train' if i<int(.8*n) else 'validation' if i<int(.9*n) else 'test') for i,g in enumerate(groups)}
    counts=collections.Counter(w for d in docs if mapping[d['protocol']]=='train' for b in d['blocks'] for w in b.split())
    vocab=['<unk>']+[w for w,c in counts.most_common(max_vocab-1) if c>=3]
    lookup={w:i for i,w in enumerate(vocab)}
    # Fixed windows never cross finding or report boundaries. No padding required.
    import numpy as np
    arrays={}; stats={}; manifest=[]
    db=sqlite3.connect(out/'auditscribe.sqlite')
    db.executescript('DROP TABLE IF EXISTS documents; DROP TABLE IF EXISTS passages; CREATE TABLE documents(path TEXT PRIMARY KEY,protocol TEXT,auditor TEXT,url TEXT,sha256 TEXT,split TEXT); CREATE TABLE passages(document_path TEXT,ordinal INTEGER,text TEXT,FOREIGN KEY(document_path) REFERENCES documents(path));')
    for d in docs:
        split=mapping[d['protocol']]
        row={k:v for k,v in d.items() if k!='blocks'}; row['split']=split; manifest.append(row)
        db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?)', tuple(row[k] for k in ['path','protocol','auditor','url','sha256','split']))
        db.executemany('INSERT INTO passages VALUES (?,?,?)',[(d['path'],i,b) for i,b in enumerate(d['blocks'])])
    for split in ['train','validation','test']:
        windows=[]; total=0; unknown=0
        for d in docs:
            if mapping[d['protocol']]!=split: continue
            for b in d['blocks']:
                ids=[lookup.get(w,0) for w in b.split()]; total+=len(ids); unknown+=ids.count(0)
                windows.extend(ids[i-context:i+1] for i in range(context,len(ids),3))
        potential=len(windows)
        limit=max_train if split=='train' else 10000
        if potential>limit:
            windows=random.Random(seed).sample(windows,limit)
        arr=np.asarray(windows,dtype=np.int64)
        if arr.ndim!=2 or len(arr)==0: raise ValueError('Empty split')
        arrays[split]=arr
        stats[split]=dict(reports=sum(mapping[d['protocol']]==split for d in docs),protocol_groups=sum(s==split for s in mapping.values()),tokens=total,unknown_fraction=unknown/total,available_windows=potential,used_windows=len(arr))
    db.commit();db.close()
    np.savez_compressed(out/'windows.npz',**arrays)
    (out/'vocab.json').write_text(json.dumps(vocab))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary=dict(source=SOURCE,revision=REVISION,seed=seed,context=context,stride=3,vocabulary_size=len(vocab),reports=len(docs),passages=sum(len(d['blocks']) for d in docs),splits=stats)
    (out/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='data/source');p.add_argument('--out',default='data/processed');p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.download: acquire(a.source)
    prepare(Path(a.source),a.out)
