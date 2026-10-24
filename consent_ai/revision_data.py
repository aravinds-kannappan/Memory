from pathlib import Path
import json,hashlib,zipfile,concurrent.futures,threading,time
import numpy as np
import pandas as pd
import requests
from .sepsis import causal_features,VITAL,META

def acquire(root,out):
    root=Path(root)/'sepsis-public';root.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path('configs/sepsis_full_selection.json').read_text());local=threading.local()
    def get(f):
        p=root/f['site']/Path(f['key']).name;p.parent.mkdir(exist_ok=True)
        if not p.exists():
            if not hasattr(local,'s'):local.s=requests.Session()
            for attempt in range(4):
                try:
                    r=local.s.get('https://physionet-open.s3.amazonaws.com/'+f['key'],timeout=60);r.raise_for_status();p.write_bytes(r.content);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(attempt+1)
        assert hashlib.md5(p.read_bytes()).hexdigest()==f['etag'],str(p)
        return dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),site=f['site'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:manifest=list(ex.map(get,spec['files']))
    Path(out,'data_manifest.json').write_text(json.dumps(manifest));return root

def features(raw,deny=None,revoke=None):
    """At revocation, clear all historical features from the revoked sources.
    Replay prefix under current permissions; do not retain pre-revocation counts.
    """
    full=causal_features(raw)
    if deny:
        masked=raw.copy();masked[list(deny)]=np.nan
        restricted=causal_features(masked)
        if revoke is None:full=restricted
        else:full.iloc[int(revoke):]=restricted.iloc[int(revoke):].to_numpy()
    return full

def load(root,out):
    cache=Path(root)/'revision-features.npz';namespath=Path(root)/'revision-columns.json'
    if cache.exists():
        z=np.load(cache,allow_pickle=False);items=[{k:z[k][i].item() for k in ['patient','site','age','gender','start','end']} for i in range(len(z['patient']))]
        return z['X'],z['y'],items,json.loads(namespath.read_text())
    xs=[];ys=[];items=[];offset=0
    for site in ['A','B']:
        for j,p in enumerate(sorted(Path(root,site).glob('*.psv'))):
            raw=pd.read_csv(p,sep='|');x=features(raw);n=len(raw)
            xs.append(x.to_numpy());ys.append(raw.SepsisLabel.to_numpy(np.int8));items.append(dict(patient=p.stem,site=site,age=float(raw.Age.iloc[0]),gender=int(raw.Gender.iloc[0]),start=offset,end=offset+n));offset+=n
            if j%3000==0:print('FEATURE_PROGRESS',site,j,flush=True)
    X=np.concatenate(xs);y=np.concatenate(ys);columns=list(x.columns)
    np.savez(cache,X=X,y=y,**{k:np.array([p[k] for p in items]) for k in items[0]});namespath.write_text(json.dumps(columns))
    return X,y,items,columns

def split(items,seed):
    rng=np.random.default_rng(seed);result={}
    for site in ['A','B']:
        ix=np.array([i for i,p in enumerate(items) if p['site']==site]);rng.shuffle(ix)
        if site=='A':
            n=len(ix);a,b,c=int(.55*n),int(.70*n),int(.85*n)
            result.update(train=ix[:a],validation=ix[a:b],calibration=ix[b:c],test_A=ix[c:])
        else:result.update(calibration_B=ix[:len(ix)//4],test_B=ix[len(ix)//4:])
    return result

def rows(items,ids):return np.concatenate([np.arange(items[i]['start'],items[i]['end']) for i in ids])
def sequences(items,ids,y,pred):
    result=[];offset=0
    for i in ids:
        p=items[i];n=p['end']-p['start'];result.append({**p,'y':y[p['start']:p['end']],'score':pred[offset:offset+n]});offset+=n
    assert offset==len(pred);return result

def preonset_rows(items,ids,y):
    selected=[]
    for i in ids:
        p=items[i];local=y[p['start']:p['end']];positive=np.flatnonzero(local)
        stop=min(len(local),int(positive[0]+6)) if len(positive) else len(local)
        selected.append(np.arange(p['start'],p['start']+stop))
    return np.concatenate(selected)
