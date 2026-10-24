from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from .sepsis import VITAL,META
from .revision_models import predict,SequenceModel
from .revision_data import rows,sequences
from .joint_risk import incident_matrices,certify

def consent_scores(est,X,y,items,ids,columns,rate,seed,revocation):
    rng=np.random.default_rng(seed);parts=[];local=[];offset=0
    vital=[j for j,c in enumerate(columns) if any(c.endswith('_'+v) for v in VITAL)]
    labs=[j for j,c in enumerate(columns) if j not in vital and not any(c=='last_'+v for v in META)]
    for i in ids:
        p=items[i];x=X[p['start']:p['end']].copy();deny=[]
        if rng.random()<rate:deny+=labs
        if rng.random()<rate:deny+=vital
        start=12 if revocation else 0
        for j in deny:x[start:,j]=0 if columns[j].startswith('count_') else np.nan
        # Reset recurrent state at withdrawal, clearing all latent cached information.
        segments=[(0,min(start,len(x))),(min(start,len(x)),len(x))] if revocation and deny else [(0,len(x))]
        for a,b in segments:
            if b>a:parts.append(x[a:b]);local.append({**p,'start':offset,'end':offset+b-a});offset+=b-a
    xx=np.concatenate(parts);return predict(est,xx,local,np.arange(len(local)))

def consent_experiment(est,X,y,items,split,columns,thresholds,cfg,seed,model):
    records=[]
    for mode in ['opt_out','revocation_12h']:
        for rate in cfg['consent_rates']:
            mats={}
            for key in ['calibration','test_A','test_B']:
                pred=consent_scores(est,X,y,items,split[key],columns,rate,seed,mode!='opt_out');seq=sequences(items,split[key],y,pred);mats[key]=incident_matrices(seq,thresholds)
            l,w,z,_=mats['calibration'];c=certify(l,w,z,budget=.5);j=c['choice']
            empirical=l[z].mean(0);valid=empirical<=.05;ej=int(np.flatnonzero(valid)[np.argmin(w.mean(0)[valid])]) if valid.any() else int(np.argmin(empirical))
            if j is not None:ej=j
            for site in ['A','B']:
                tl,tw,tz,_=mats['test_'+site]
                rec=dict(seed=seed,model=model,site=site,mode=mode,optout_rate=rate,status=c['status'],certified_threshold=None if j is None else float(thresholds[j]),empirical_selected_threshold=float(thresholds[ej]),test_risk=float(tl[tz,ej].mean()),reviews_per_patient=float(tw[:,ej].mean()),selection='empirical when no certificate; not a guarantee',guarantee_applies=site=='A' and j is not None)
                records.append(rec)
    return records

def queues(seq,threshold,cfg,seed,model,site):
    loss,review,pos,first=incident_matrices(seq,[threshold]);rng=np.random.default_rng(seed);rows=[]
    for arrival in cfg['patient_arrivals_per_hour']:
        admission=np.cumsum(rng.exponential(1/arrival,len(seq)));requests=[]
        for i,p in enumerate(seq):
            if review[i,0]:requests.append((admission[i]+first[i,0],i))
        requests.sort()
        for minutes in cfg['service_minutes']:
            for sensitivity in cfg['review_sensitivity']:
                for specificity in cfg['review_specificity']:
                    end=0.;miss=np.ones(len(seq),bool);waits=[];fp=0;finished=0
                    for at,i in requests:
                        wait=max(0,end-at);service=minutes/60;end=max(end,at)+service;waits.append(wait);finished+=1
                        # Sensitivity degrades with waiting, a declared sensitivity analysis.
                        fatigue=max(.5,1-.05*wait);correct=rng.random()<sensitivity*fatigue
                        if pos[i]:
                            onset=np.flatnonzero(seq[i]['y'])[0]+6
                            if not loss[i,0] and first[i,0]+wait+service<onset and correct:miss[i]=False
                        elif rng.random()>specificity:fp+=1
                    rows.append(dict(seed=seed,model=model,site=site,threshold=threshold,patient_arrivals_per_hour=arrival,service_minutes=minutes,reviewer_sensitivity=sensitivity,reviewer_specificity=specificity,reviews=finished,patients=len(seq),mean_wait_minutes=float(np.mean(waits)*60) if waits else 0,p95_wait_minutes=float(np.quantile(waits,.95)*60) if waits else 0,end_to_end_conditional_miss=float(miss[pos].mean()),false_actions=fp,source='SIMULATION ON REAL SEPSIS SCORES',fatigue='sensitivity times max(.5,1-.05*waiting_hours)'))
    return rows

def random_graphs(cfg,out):
    rng=np.random.default_rng(20261024);rows=[]
    for trial in range(cfg['graph_replicates']):
        n=int(rng.integers(50,301));g=nx.DiGraph();g.add_nodes_from(range(n))
        for i in range(n-1):
            for j in range(i+1,n):
                if rng.random()<3/n:g.add_edge(i,j)
        h=set(rng.choice(np.arange(1,n-1),max(1,n//10),replace=False).tolist());q=g.copy();q.remove_nodes_from(h)
        reach=nx.has_path(g,0,n-1);bypass=nx.has_path(q,0,n-1)
        rows.append(dict(trial=trial,nodes=n,edges=g.number_of_edges(),review_nodes=len(h),action_reachable=reach,unreviewed_path=bypass,source='RANDOM DAG SIMULATION'))
    pd.DataFrame(rows).to_csv(Path(out,'graph_ensemble.csv'),index=False)
