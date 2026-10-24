"""Public inputs, deterministic manifests and credentialed schema readers."""
import hashlib, json, re, zipfile, io
from pathlib import Path
import requests
import pandas as pd
import numpy as np

MIMIC_FILES=['hosp/patients.csv.gz','hosp/admissions.csv.gz','hosp/labevents.csv.gz','hosp/prescriptions.csv.gz','hosp/poe.csv.gz','icu/icustays.csv.gz','icu/chartevents.csv.gz','icu/inputevents.csv.gz','icu/d_items.csv.gz','SHA256SUMS.txt','LICENSE.txt']
SKELETON_MD5={'SUBJECT1.zip':'6c34ff6e898b42ecead633836206a5b0','SUBJECT2.zip':'4db506e5a541ad2de84bf5009d440f13','SUBJECT3.zip':'5f3d6b4a675491904afb002b7e75502b','SUBJECT4.zip':'76a641d4bb79488a0563dde1041f95c8','SUBJECT5.zip':'2676346b213d6e0f4da6fda09f0b2b92'}

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.part')
        with requests.get(url,stream=True,timeout=(20,120)) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
        tmp.replace(path)
    return path

def sha(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def acquire(root,out):
    root=Path(root);manifest=[]
    # User approved substituting openly downloadable derived skeletons.
    for name,md5 in SKELETON_MD5.items():
        url=f'https://zenodo.org/records/12773013/files/{name}?download=1'
        p=download(url,root/'upfall-skeleton'/name)
        assert sha(p,'md5')==md5, f'Publisher checksum mismatch: {name}'
        manifest.append(dict(dataset='UP-Fall skeleton derivative',path=str(p),url=url,sha256=sha(p),bytes=p.stat().st_size,license='CC BY 4.0',publisher_checksum_verified=True))
    for name in MIMIC_FILES:
        url='https://physionet.org/files/mimic-iv-demo/2.2/'+name
        p=download(url,root/'mimic-demo'/name)
        manifest.append(dict(dataset='MIMIC-IV DEMO DATA v2.2',path=str(p),url=url,sha256=sha(p),bytes=p.stat().st_size,license='PhysioNet Open Access'))
    checks={line.split()[-1].lstrip('./'):line.split()[0] for line in (root/'mimic-demo/SHA256SUMS.txt').read_text().splitlines() if line.strip()}
    for name in MIMIC_FILES:
        if name in checks:assert sha(root/'mimic-demo'/name)==checks[name],name
    Path(out,'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def table(root,module,name,columns=None,chunksize=None):
    root=Path(root);p=root/module/(name+'.csv.gz')
    if not p.exists():p=root/module/(name+'.csv')
    return pd.read_csv(p,usecols=columns,chunksize=chunksize,low_memory=False)

def upfall_skeleton(root):
    records=[];audit=[]
    for path in sorted(Path(root).glob('SUBJECT*.zip')):
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if not name.endswith('.csv'):continue
                # One source filename lacks its camera prefix; do not invent a camera.
                m=re.search(r'S(\d+)_?A(\d+)_?T(\d+)',name)
                if m is None:raise ValueError(f'Unknown filename {name}')
                subject,activity,trial=map(int,m.groups());d=pd.read_csv(z.open(name))
                expected=[f'Joint{j}_{axis}' for j in range(1,34) for axis in 'XYZ']
                if list(d.columns[:99])!=expected or d.shape[1]!=100:raise ValueError(f'Unexpected coordinate schema {name}')
                y=d.iloc[:,-1].to_numpy();assert np.isin(y,[0,1]).all()
                x=d[expected].to_numpy(dtype=np.float32).reshape(-1,33,3)
                assert np.isfinite(x).all()
                camera=re.match(r'C(\d+)',name)
                audit.append(dict(file=name,subject=subject,trial=trial,activity=activity,frames=len(x),label_column=d.columns[-1],positive_frames=int(y.sum()),sha256=hashlib.sha256(z.read(name)).hexdigest(),camera=int(camera.group(1)) if camera else None))
                records.append(dict(file=name,subject=subject,trial=trial,activity=activity,x=x,y=y.astype(int)))
    return records,audit

def ntu_skeleton(path):
    """Official .skeleton format: frames, bodies, body info, joint count, joint rows.

    Returns T x 25 x 3; primary body is selected by most observed frames.
    Missing body frames are NaN and must be handled within the observed prefix.
    """
    with open(path) as f:
        frames=int(f.readline());tracks={}
        for t in range(frames):
            for _ in range(int(f.readline())):
                info=f.readline().split();body=info[0];nj=int(f.readline())
                if nj!=25:raise ValueError('Expected 25 NTU joints')
                xyz=np.array([list(map(float,f.readline().split()[:3])) for _ in range(nj)])
                tracks.setdefault(body,{})[t]=xyz
    if not tracks:raise ValueError('No body tracked')
    body=max(tracks,key=lambda b:len(tracks[b]));x=np.full((frames,25,3),np.nan,dtype=np.float32)
    for t,xyz in tracks[body].items():x[t]=xyz
    m=re.search(r'S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)',Path(path).name)
    if not m:raise ValueError('Expected NTU S/C/P/R/A filename')
    return {'x':x,**dict(zip(['setup','camera','subject','replication','activity'],map(int,m.groups())))}

def safer_annotations(path):
    """Read public documented annotation schema without guessing subject IDs."""
    d=pd.read_csv(path)
    required={'Action','Start Time','End Time'}
    if not required.issubset(d.columns):raise ValueError('SAFER annotation schema mismatch')
    d['start_seconds']=pd.to_timedelta(d['Start Time']).dt.total_seconds()
    d['end_seconds']=pd.to_timedelta(d['End Time']).dt.total_seconds()
    assert (d.end_seconds>=d.start_seconds).all()
    return d
