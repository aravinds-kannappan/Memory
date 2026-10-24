from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .sepsis import auc_structure,weighted_auc
from .joint_risk import incident_matrices,certify,monotone_crc
from .vendor.evaluate_sepsis_score import compute_prediction_utility

def utility_curve(seq,thresholds,once):
    observed=np.zeros(len(thresholds));inaction=0.;best=0.
    for p in seq:
        y=np.asarray(p['y']);s=np.asarray(p['score']);n=len(y);zeros=np.zeros(n,int);ones=np.ones(n,int)
        # The official utility is additive: recover its per-hour contribution.
        where=np.flatnonzero(y);onset=int(where[0]+6) if len(where) else np.inf;t=np.arange(n);dt=t-onset
        u1=np.where(dt<=-6,np.maximum((dt+12)/6,-.05),np.where(dt<=3,1-(dt+6)/9,0.)) if len(where) else np.full(n,-.05)
        u0=np.where((dt>-6)&(dt<=3),-2*(dt+6)/9,0.) if len(where) else np.zeros(n)
        # Match official post-window FN contribution exactly.
        if len(where):u0[dt>3]=0.
        base=float(u0.sum());assert np.isclose(base,compute_prediction_utility(y,zeros))
        assert np.isclose(u1.sum(),compute_prediction_utility(y,ones))
        decisions=s[:,None]>=np.asarray(thresholds)[None,:]
        if once:decisions=np.maximum.accumulate(decisions,axis=0) # latched clinical state; one review
        observed+=base+((u1-u0)[:,None]*decisions).sum(0);inaction+=base
        optimal=np.zeros(n);optimal[max(0,int(onset-12)) if np.isfinite(onset) else 0:min(n,int(onset+4)) if np.isfinite(onset) else 0]=1
        best+=compute_prediction_utility(y,optimal)
    return (observed-inaction)/(best-inaction) if best>inaction else np.full(len(thresholds),np.nan)

def discrimination(seq,reps,seed):
    seq=[{**p,'y':p['y'][:min(len(p['y']),int(np.flatnonzero(p['y'])[0]+6))] if np.any(p['y']) else p['y'],'score':p['score'][:min(len(p['y']),int(np.flatnonzero(p['y'])[0]+6))] if np.any(p['y']) else p['score']} for p in seq]
    y=np.concatenate([p['y'] for p in seq]);s=np.concatenate([p['score'] for p in seq]);patient=np.repeat(np.arange(len(seq)),[len(p['y']) for p in seq]);structure=auc_structure(y,s);rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(reps):
        counts=rng.multinomial(len(seq),np.full(len(seq),1/len(seq)));w=counts[patient]
        vals.append([weighted_auc(structure,w),np.average((y-s)**2,weights=w)])
    arr=np.array(vals);low,high=np.nanquantile(arr,[.025,.975],axis=0)
    return dict(auroc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)),brier=float(np.mean((y-s)**2)),auroc_lower=low[0],auroc_upper=high[0],brier_lower=low[1],brier_upper=high[1],bootstrap_replicates=reps,patients=len(seq),hours=len(y))

def curves(seq,thresholds,seed,model,site,policy,with_utility=True):
    loss,review,pos,first=incident_matrices(seq,thresholds,once=policy=='once')
    utilities=utility_curve(seq,thresholds,once=policy=='once') if with_utility else np.full(len(thresholds),np.nan)
    risk=loss[pos].mean(0) if pos.any() else np.full(len(thresholds),np.nan)
    if policy=='hourly':workload=np.array([np.mean([np.sum(p['score'][:min(len(p['y']),int(np.argmax(p['y'])+6) if np.any(p['y']) else len(p['y']))]>=t) for p in seq]) for t in thresholds])
    else:workload=review.mean(0)
    result=[dict(seed=seed,model=model,site=site,policy=policy,threshold=float(t),conditional_incident_miss=float(risk[j]),reviews_per_patient=float(workload[j]),any_review_fraction=float(review[:,j].mean()),challenge_utility=float(utilities[j]),positive_incident_patients=int(pos.sum()),patients=len(seq)) for j,t in enumerate(thresholds)]
    return result,(loss,review,pos,first)

def policy_analysis(seq_by_split,thresholds,cfg,seed,model,out):
    rows=[];certs=[];group_rows=[];draws=[];site_curves={};matrices={}
    for key in ['calibration','calibration_B','test_A','test_B']:
        matrices[key]=incident_matrices(seq_by_split[key],thresholds)
    for site in ['A','B']:
        for policy in ['once','hourly']:
            r,mat=curves(seq_by_split['test_'+site],thresholds,seed,model,site,policy);rows+=r
            if policy=='once':site_curves[site]=r
        for origin in (['A'] if site=='A' else ['A','B']):
            calkey='calibration' if origin=='A' else 'calibration_B';loss,review,pos,_=matrices[calkey]
            for budget in cfg['budgets']:
                c=certify(loss,review,pos,cfg['alpha'],budget,cfg['delta']);j=c['choice'];row=dict(seed=seed,model=model,site=site,calibration_site=origin,budget=budget,status=c['status'],n_positive=c['n_positive'],n=c['n'],threshold=None if j is None else float(thresholds[j]),guarantee_applies=site==origin)
                if j is not None:row.update(site_curves[site][j],risk_upper=float(c['risk_upper'][j]),workload_upper=float(c['workload_upper'][j]))
                certs.append(row);print('JOINT_CERTIFICATE',json.dumps(row),flush=True)
        for group,fn in [('female',lambda p:p['gender']==0),('male',lambda p:p['gender']==1),('age_under65',lambda p:p['age']<65),('age65plus',lambda p:p['age']>=65)]:
            calkey='calibration' if site=='A' else 'calibration_B';cm=np.array([fn(p) for p in seq_by_split[calkey]]);tm=np.array([fn(p) for p in seq_by_split['test_'+site]])
            l,w,z,_=matrices[calkey];c=certify(l[cm],w[cm],z[cm],budget=.5,delta=cfg['delta']/4);j=c['choice'];tl,tw,tz,_=matrices['test_'+site]
            group_rows.append(dict(seed=seed,model=model,site=site,group=group,status=c['status'],n_positive=c['n_positive'],test_positive=int((tz&tm).sum()),risk=np.mean(tl[tz&tm,j]) if j is not None and (tz&tm).any() else None,workload=np.mean(tw[tm,j]) if j is not None else None))
    # Repeated random calibration subsets from a fixed pool; dependence across draws is explicit.
    rng=np.random.default_rng(seed)
    for site in ['A','B']:
        ck='calibration' if site=='A' else 'calibration_B';l,w,z,_=matrices[ck];tl,tw,tz,_=matrices['test_'+site];positive=np.flatnonzero(z)
        for n in cfg['positive_calibration_sizes_B']:
            if n>len(positive):continue
            for draw in range(cfg['calibration_draws']):
                sub=rng.choice(positive,n,replace=False)
                # Incident-positive sampling targets conditional risk; workload uses independent all-patient calibration pool.
                from .joint_risk import binomial_bounds
                _,upper=binomial_bounds(l[sub].sum(0),n,cfg['delta']/(2*len(thresholds)))
                ok=upper<=cfg['alpha'];j=int(np.flatnonzero(ok)[np.argmin(w.mean(0)[ok])]) if ok.any() else None
                risk=float(tl[tz,j].mean()) if j is not None else None
                draws.append(dict(seed=seed,model=model,site=site,n_positive=n,draw=draw,certified=j is not None,test_risk=risk,exceeds_alpha=None if risk is None else risk>cfg['alpha'],reviews_per_patient=float(tw[:,j].mean()) if j is not None else None))
    return rows,certs,group_rows,draws,matrices
