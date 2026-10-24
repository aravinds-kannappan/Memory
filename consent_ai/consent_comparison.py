"""Paired inference-time consent interventions on frozen sepsis predictors.

Training, patients, labels and decision thresholds remain fixed. No clinical
non-inferiority margin is selected from test results. Full inputs are a public-
benchmark reference condition, not permission to collect data without consent.
"""
from pathlib import Path
import hashlib,json,os,pickle,time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from . import revision_data as data, revision_models as models
from .sepsis import VITAL,META,auc_structure,weighted_auc
from .joint_risk import incident_matrices
from .vendor.evaluate_sepsis_score import compute_prediction_utility


def groups(columns):
    vital=[j for j,c in enumerate(columns) if any(c.endswith('_'+v) for v in VITAL)]
    lab=[j for j,c in enumerate(columns) if j not in vital and c not in ['last_'+v for v in META]]
    return lab,vital


def permission_draw(seed,patient,site):
    digest=hashlib.sha256(f'{seed}:{site}:{patient}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'little')).random(2)


def restricted_predict(est,X,items,ids,columns,scenario,seed):
    lab,vital=groups(columns);last=[j for j,c in enumerate(columns) if c.startswith('last_') and j in lab+vital]
    parts=[];segments=[];offset=0;exposure=[]
    for i in ids:
        p=items[i];x=X[p['start']:p['end']].copy();n=len(x);u=permission_draw(seed,p['patient'],p['site'])
        deny=(lab if u[0]<scenario['lab_denial'] else [])+(vital if u[1]<scenario['vital_denial'] else [])
        start=min(scenario['start'],n);original=np.isfinite(x[:,last]).sum()
        for j in deny:x[start:,j]=0 if columns[j].startswith('count_') else np.nan
        kept=np.isfinite(x[:,last]).sum();denied_last=len(set(deny)&set(last))
        exposure.append([int(bool(deny)),n*len(last),n*len(last)-(n-start)*denied_last,original,kept])
        cuts=[(0,start),(start,n)] if deny and start>0 else [(0,n)]
        for a,b in cuts:
            if b>a:parts.append(x[a:b]);segments.append({**p,'start':offset,'end':offset+b-a});offset+=b-a
    score=models.predict(est,np.concatenate(parts),segments,np.arange(len(segments)))
    assert len(score)==sum(items[i]['end']-items[i]['start'] for i in ids)
    e=np.asarray(exposure,dtype=float)
    return score,dict(denied_patient_fraction=float(e[:,0].mean()),permitted_source_hour_fraction=float(e[:,2].sum()/e[:,1].sum()),available_feature_cell_fraction=float(e[:,4].sum()/e[:,3].sum()))


def preonset(seq):
    ys=[];scores=[];owners=[]
    for i,p in enumerate(seq):
        pos=np.flatnonzero(p['y']);stop=min(len(p['y']),int(pos[0]+6)) if len(pos) else len(p['y'])
        ys.append(p['y'][:stop]);scores.append(p['score'][:stop]);owners.extend([i]*stop)
    return np.concatenate(ys),np.concatenate(scores),np.asarray(owners)


def paired_discrimination(y,base,other,owners,n,reps,seed):
    bs=auc_structure(y,base);cs=auc_structure(y,other);rng=np.random.default_rng(seed);draws=[]
    for _ in range(reps):
        count=rng.multinomial(n,np.full(n,1/n));w=count[owners]
        draws.append([weighted_auc(cs,w)-weighted_auc(bs,w),np.average((y-other)**2-(y-base)**2,weights=w)])
    lo,hi=np.nanquantile(draws,[.025,.975],axis=0)
    result={}
    for metric,fn in [('auroc',roc_auc_score),('average_precision',average_precision_score)]:
        a=float(fn(y,base));b=float(fn(y,other));result.update({metric:b,'full_'+metric:a,'delta_'+metric:b-a})
    result.update(brier=float(np.mean((y-other)**2)),full_brier=float(np.mean((y-base)**2)),delta_brier=float(np.mean((y-other)**2-(y-base)**2)),delta_auroc_lower=float(lo[0]),delta_auroc_upper=float(hi[0]),delta_brier_lower=float(lo[1]),delta_brier_upper=float(hi[1]))
    return result


def policy_vectors(seq,threshold,once):
    loss,review,pos,_=incident_matrices(seq,[threshold],once=once)
    work=[];observed=[];inaction=[];optimal=[];false=[]
    for p in seq:
        y=p['y'];score=p['score'];n=len(y);positive=np.flatnonzero(y);onset=int(positive[0]+6) if len(positive) else None;stop=min(n,onset) if onset is not None else n
        decisions=(score>=threshold).astype(int)
        work.append(int(decisions[:stop].any()) if once else int(decisions[:stop].sum()))
        false.append((int(decisions.any()) if once else int(decisions.sum())) if onset is None else 0)
        if once:decisions=np.maximum.accumulate(decisions)
        best=np.zeros(n,int)
        if onset is not None:best[max(0,onset-12):min(n,onset+4)]=1
        observed.append(compute_prediction_utility(y,decisions));inaction.append(compute_prediction_utility(y,np.zeros(n,int)));optimal.append(compute_prediction_utility(y,best))
    return dict(loss=loss[:,0].astype(float),positive=pos,work=np.asarray(work,float),observed=np.asarray(observed),inaction=np.asarray(inaction),optimal=np.asarray(optimal),false=np.asarray(false,float),negative=np.asarray([not np.any(p['y']) for p in seq]))


def policy_metrics(v,weights):
    positive=weights*v['positive'];negative=weights*v['negative'];den=np.dot(weights,v['optimal']-v['inaction'])
    return np.array([np.dot(positive,v['loss'])/positive.sum(),np.dot(weights,v['work'])/weights.sum(),np.dot(weights,v['observed']-v['inaction'])/den,np.dot(weights,v['false'])/negative.sum()])


def paired_policy(base,other,reps,seed):
    n=len(base['work']);one=np.ones(n);a=policy_metrics(base,one);b=policy_metrics(other,one);rng=np.random.default_rng(seed);vals=[]
    for _ in range(reps):
        w=rng.multinomial(n,np.full(n,1/n));vals.append(policy_metrics(other,w)-policy_metrics(base,w))
    lo,hi=np.quantile(vals,[.025,.975],axis=0);out={}
    for j,name in enumerate(['risk','reviews','utility','false_reviews_per_negative']):
        out.update({name:float(b[j]),'full_'+name:float(a[j]),'delta_'+name:float(b[j]-a[j]),'delta_'+name+'_lower':float(lo[j]),'delta_'+name+'_upper':float(hi[j])})
    return out


def load_model(path,model,seed,d):
    if model!='gru':
        with (path/f'{model}-{seed}.pkl').open('rb') as f:return pickle.load(f)
    state=torch.load(path/f'gru-{seed}.pt',map_location='cpu',weights_only=False);net=models.GRU(d,state['hidden']);net.load_state_dict(state['state_dict'])
    return models.SequenceModel(net,state['mean'],state['scale'])


def run(root,output):
    start=time.time();cfg=json.loads(Path('configs/consent_comparison.json').read_text());base=Path(output)/'revision';audit=Path(output)/'revision-audit';out=Path(output)/'consent-comparison';out.mkdir(parents=True,exist_ok=True)
    parent=json.loads((base/'provenance.json').read_text());prov={'run_id':os.getenv('ORX_RUN_ID'),'commit':os.getenv('ORX_COMMIT'),'configuration':cfg,'source_run':parent['run_id'],'inputs':[]}
    print('CONSENT_COMPARISON_CONFIGURATION',json.dumps(prov),flush=True)
    if not (base/'cohort.csv').exists():raise FileNotFoundError('Run the full revision and audit stages before the paired comparison.')
    root=Path(root)/'sepsis-public';X,y,items,columns=data.load(root,out);torch.set_num_threads(2)
    selected=pd.read_csv(audit/'selected_policies.csv');metrics=[];policies=[]
    for seed in cfg['seeds']:
        split=data.split(items,seed)
        for model in cfg['models']:
            est=load_model(base,model,seed,X.shape[1]);path=base/f'{model}-{seed}.{ "pt" if model=="gru" else "pkl"}'
            prov['inputs'].append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for site in cfg['sites']:
                ids=split['test_'+site];pfile=base/f'predictions-{model}-{seed}-test_{site}.npz';saved=np.load(pfile);assert np.array_equal(ids,saved['ids'])
                prov['inputs'].append({'file':pfile.name,'sha256':hashlib.sha256(pfile.read_bytes()).hexdigest()})
                # Recompute baseline through exactly the same prediction path as interventions.
                baseline,exp=restricted_predict(est,X,items,ids,columns,cfg['scenarios'][0],seed)
                assert np.allclose(baseline,saved['score'],atol=2e-6), 'Saved full-input scores changed'
                seq0=data.sequences(items,ids,y,baseline);yy,ss,owners=preonset(seq0)
                thresholds={p:float(selected[(selected.seed==seed)&(selected.model==model)&(selected.site==site)&(selected.policy==p)].iloc[0].threshold) for p in ['once','hourly']}
                pv0={p:policy_vectors(seq0,t,p=='once') for p,t in thresholds.items()}
                for scenario in cfg['scenarios']:
                    name=scenario['name'];score,exposure=(baseline,exp) if name=='full_input' else restricted_predict(est,X,items,ids,columns,scenario,seed)
                    seq=data.sequences(items,ids,y,score);yyy,s,own=preonset(seq);assert np.array_equal(yyy,yy) and np.array_equal(own,owners)
                    common=dict(seed=seed,model=model,site=site,scenario=name,patients=len(ids),preonset_hours=len(yy),**exposure)
                    r={**common,**paired_discrimination(yy,ss,s,owners,len(ids),cfg['discrimination_bootstrap'],seed)};metrics.append(r);print('PAIRED_DISCRIMINATION',json.dumps(r),flush=True)
                    for p,t in thresholds.items():
                        vectors=pv0[p] if name=='full_input' else policy_vectors(seq,t,p=='once')
                        result={**common,'policy':p,'threshold':t,**paired_policy(pv0[p],vectors,cfg['policy_bootstrap'],seed)};policies.append(result)
                        np.savez_compressed(out/f'patient-effects-{model}-{seed}-{site}-{name}-{p}.npz',patient=np.asarray([items[i]['patient'] for i in ids]),**{k:vectors[k]-pv0[p][k] for k in ['loss','work','observed','false']},positive=vectors['positive'],negative=vectors['negative'],utility_denominator=vectors['optimal']-vectors['inaction'])
                    pd.DataFrame(metrics).to_csv(out/'discrimination.csv',index=False);pd.DataFrame(policies).to_csv(out/'policies.csv',index=False)
                    print('CONSENT_SCENARIO_COMPLETE',seed,model,site,name,flush=True)
    (out/'provenance.json').write_text(json.dumps(prov,indent=2));summary={'discrimination_rows':len(metrics),'policy_rows':len(policies),'patients':len(items),'models':len(cfg['models']),'seeds':cfg['seeds'],'mean_deltas':pd.DataFrame(metrics).groupby(['scenario','model','site'])[['delta_auroc','delta_brier','permitted_source_hour_fraction']].mean().reset_index().to_dict('records'),'clinical_noninferiority_claim':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('PAIRED_CONSENT_SUMMARY',json.dumps(summary),flush=True);print('CONSENT_COMPARISON_COMPLETE',json.dumps({'run_id':prov['run_id'],'seconds':time.time()-start,'metrics':len(metrics),'policies':len(policies)}),flush=True)
