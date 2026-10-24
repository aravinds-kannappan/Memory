"""Re-evaluate selected policies and reviewer errors from frozen predictions."""
from pathlib import Path
import json, os, hashlib
import numpy as np
import pandas as pd
from .revision_data import sequences
from .revision_evaluation import utility_curve
from .joint_risk import incident_matrices, binomial_bounds
from .revision_stress import queues

def run(root, output):
    base=Path(output)/'revision';out=Path(output)/'revision-audit';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('configs/revision_protocol.json').read_text());cohort=pd.read_csv(base/'cohort.csv').replace({np.nan:None}).to_dict('records')
    cache=np.load(Path(root)/'sepsis-public'/'revision-features.npz',allow_pickle=True);y=cache['y']
    print('AUDIT_CONFIGURATION',json.dumps({'run_id':os.getenv('ORX_RUN_ID'),'commit':os.getenv('ORX_COMMIT'),'parent':json.loads((base/'provenance.json').read_text()),'policy_selection':'validation challenge utility; no test tuning','queue_correction':'false actions only on patients with no positive sepsis label'}),flush=True)
    metrics=[];queue_rows=[];evidence=[]
    for seed in cfg['seeds']:
        for model in cfg['models']:
            thresholds=np.array(json.loads((base/f'thresholds-{model}-{seed}.json').read_text()));seq={}
            for split in ['validation','test_A','test_B']:
                p=base/f'predictions-{model}-{seed}-{split}.npz';z=np.load(p);seq[split]=sequences(cohort,z['ids'],y,z['score']);evidence.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            for policy in ['once','hourly']:
                once=policy=='once';j=int(np.argmax(utility_curve(seq['validation'],thresholds,once)));t=thresholds[j]
                for site in ['A','B']:
                    ps=seq['test_'+site];loss,w,pos,first=incident_matrices(ps,[t],once=once);n=int(pos.sum());k=int(loss[pos,0].sum());low,high=binomial_bounds(k,n,.025)
                    early=sum(bool(pos[i] and first[i,0]>=0 and first[i,0]<np.flatnonzero(p['y'])[0]) for i,p in enumerate(ps))
                    noalert=int((pos&~w[:,0]).sum());risk=k/n
                    workload=float(w[:,0].mean()) if once else float(np.mean([np.sum(p['score'][:min(len(p['y']),int(np.flatnonzero(p['y'])[0]+6))] >= t) if np.any(p['y']) else np.sum(p['score']>=t) for p in ps]))
                    r=dict(seed=seed,model=model,site=site,policy=policy,threshold=float(t),positive_incident_patients=n,misses=k,risk=risk,risk_lower=float(low),risk_upper=float(high),early_first_alerts=early,no_preonset_alerts=noalert,reviews_per_patient=workload,challenge_utility=float(utility_curve(ps,[t],once)[0]),selection='validation utility',interval='pointwise exact binomial 95%; no multiplicity claim')
                    metrics.append(r);print('SELECTED_POLICY',json.dumps(r),flush=True)
                    if once:queue_rows+=queues(ps,float(t),cfg,seed,model,site)
    pd.DataFrame(metrics).to_csv(out/'selected_policies.csv',index=False);pd.DataFrame(queue_rows).to_csv(out/'queues.csv',index=False)
    provenance=dict(run_id=os.getenv('ORX_RUN_ID'),commit=os.getenv('ORX_COMMIT'),parent_run=json.loads((base/'provenance.json').read_text())['run_id'],inputs=evidence)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2));print('AUDIT_COMPLETE',json.dumps({'policies':len(metrics),'queue_scenarios':len(queue_rows),'run_id':provenance['run_id']}),flush=True)
