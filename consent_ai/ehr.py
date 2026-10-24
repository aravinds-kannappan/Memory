"""Causal landmark features and patient-disjoint EHR evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from .data import table
from .evaluation import fit_model, probability, metrics, cluster_bootstrap, calibration_bins, paired_bootstrap
from .risk import calibrate, ASSUMPTIONS, LOSS

def available(events,cutoff,start=None):
    """Require event AND storage time before cutoff; never backfill from future."""
    d=events[(events.charttime<=cutoff)&(events.storetime<=cutoff)].copy()
    if start is not None:d=d[d.charttime>=start]
    assert (d.charttime<=cutoff).all() and (d.storetime<=cutoff).all()
    return d

def read_events(root,module,name,items,key):
    chunks=[]
    cols=[key,'subject_id','itemid','charttime','storetime','valuenum']
    for d in table(root,module,name,columns=list(dict.fromkeys(cols)),chunksize=250000):
        d=d[d.itemid.isin(items)].copy()
        for c in ['charttime','storetime']:d[c]=pd.to_datetime(d[c],errors='coerce')
        chunks.append(d)
    allrows=pd.concat(chunks,ignore_index=True)
    return {k:g for k,g in allrows.groupby(key)},allrows

def summarize(events,cutoff,start,items,prefix):
    d=available(events,cutoff,start);features={}
    for item in items:
        q=d[d.itemid==item].sort_values(['charttime','storetime'])
        vals=q.valuenum.dropna()
        features[f'{prefix}{item}_last']=float(vals.iloc[-1]) if len(vals) else np.nan
        features[f'{prefix}{item}_mean']=float(vals.mean()) if len(vals) else np.nan
        features[f'{prefix}{item}_count']=len(vals)
    return features

def build(root,cfg,out):
    p=table(root,'hosp','patients');a=table(root,'hosp','admissions');s=table(root,'icu','icustays')
    for c in ['admittime','dischtime','deathtime']:a[c]=pd.to_datetime(a[c],errors='coerce')
    for c in ['intime','outtime']:s[c]=pd.to_datetime(s[c],errors='coerce')
    cohort=s.merge(a,on=['subject_id','hadm_id'],validate='many_to_one').merge(p,on='subject_id',validate='many_to_one')
    vitals,emptyv=read_events(root,'icu','chartevents',cfg['vital_items'],'stay_id')
    labs,emptyl=read_events(root,'hosp','labevents',cfg['lab_items'],'hadm_id')
    med=table(root,'icu','inputevents',columns=['subject_id','stay_id','starttime','storetime','itemid'])
    for c in ['starttime','storetime']:med[c]=pd.to_datetime(med[c],errors='coerce')
    med=med[med.itemid.isin(cfg['sedative_items'])]
    meds={k:g for k,g in med.groupby('stay_id')}
    rows=[]
    for x in cohort.itertuples():
        for h in cfg['landmarks_hours']:
            cutoff=x.intime+pd.Timedelta(hours=h)
            if cutoff>=x.outtime or cutoff>=x.dischtime or (pd.notna(x.deathtime) and cutoff>=x.deathtime):continue
            # Death after discharge is not part of this in-hospital outcome.
            y=int(pd.notna(x.deathtime) and cutoff<x.deathtime<=cutoff+pd.Timedelta(hours=cfg['horizon_hours']))
            r=dict(patient=x.subject_id,stay_id=x.stay_id,hadm_id=x.hadm_id,row_id=f'{x.stay_id}_{h}',
                   cutoff=str(cutoff),landmark_hours=h,y=y,era=x.anchor_year_group,
                   age=x.anchor_age+(x.intime.year-x.anchor_year),female=int(x.gender=='F'),
                   gender=x.gender,race=x.race)
            r.update(summarize(vitals.get(x.stay_id,emptyv.iloc[:0]),cutoff,x.intime,cfg['vital_items'],'v'))
            r.update(summarize(labs.get(x.hadm_id,emptyl.iloc[:0]),cutoff,x.admittime,cfg['lab_items'],'l'))
            m=meds.get(x.stay_id,med.iloc[:0]);r['sedative_recorded']=int(((m.starttime<=cutoff)&(m.storetime<=cutoff)).any())
            rows.append(r)
    d=pd.DataFrame(rows)
    assert not d.row_id.duplicated().any()
    d.to_csv(Path(out,'ehr_cohort.csv'),index=False)
    audit={'source':'DEMO DATA' if 'demo' in str(root) else 'CREDENTIALED DATA','raw_patients':len(p),
           'eligible_patients':d.patient.nunique(),'landmarks':len(d),'positive_landmarks':int(d.y.sum()),
           'positive_patients':d.groupby('patient').y.max().sum(),'era_counts':d.groupby('era').patient.nunique().to_dict(),
           'availability':'charttime AND storetime <= prediction cutoff','site_holdout':'unavailable: one institution',
           'temporal_holdout':'coarse anchor-era shift, not exact calendar ordering','missing_feature_fraction':d.filter(regex='^[vl][0-9]').isna().mean().to_dict()}
    Path(out,'ehr_audit.json').write_text(json.dumps(audit,indent=2,default=int))
    print('EHR_AUDIT',json.dumps(audit,default=int),flush=True)
    return d

def split_units(d,seed,regime):
    rng=np.random.default_rng(seed)
    if regime=='random':
        ids=np.array(sorted(d.patient.unique()));rng.shuffle(ids);n=len(ids)
        train,cal,test=np.split(ids,[int(.5*n),int(.75*n)])
    else:
        early=sorted(d.era.unique())[0];dev=np.array(sorted(d.loc[d.era==early,'patient'].unique()));rng.shuffle(dev)
        train,cal=np.split(dev,[int(.65*len(dev))]);test=np.array(sorted(d.loc[d.era!=early,'patient'].unique()))
    assert not (set(train)&set(cal) or set(train)&set(test) or set(cal)&set(test))
    return dict(train=train.tolist(),calibration=cal.tolist(),test=test.tolist())

def evaluate(d,cfg,out):
    predictions=[];summaries=[];intervals=[];bins=[];subgroups=[];splits=[];calibrations=[];deltas=[]
    meta=['age','female','landmark_hours'];v=list(d.filter(regex='^v[0-9]').columns);l=list(d.filter(regex='^l[0-9]').columns)
    variants={'metadata':meta,'full_ehr':meta+v+l+['sedative_recorded'],'consent_no_labs':meta+v+['sedative_recorded'],'consent_no_vitals':meta+l+['sedative_recorded']}
    for seed in cfg['seeds']:
        for regime in ['random','era_shift']:
            split=split_units(d,seed,regime);splits.append(dict(seed=seed,regime=regime,**split))
            train=d[d.patient.isin(split['train'])];cal=d[d.patient.isin(split['calibration'])];test=d[d.patient.isin(split['test'])]
            paired={}
            for name,cols in variants.items():
                model=fit_model(train[cols],train.y);cs=probability(model,cal[cols]);ts=probability(model,test[cols])
                crc=calibrate(cal.y,cs,cal.patient,cfg['alpha']);curve=crc.pop('curve');calibrations.append(dict(seed=seed,regime=regime,model=name,**crc))
                z=test[['patient','row_id','y','gender','race','age','landmark_hours']].copy();z['score']=ts;z['lambda']=crc['lambda'];z['seed']=seed;z['regime']=regime;z['model']=name;z['source']='DEMO DATA'
                predictions.append(z);paired[name]=z
                for policy,lam in [('fixed_0.5',.5),('crc',crc['lambda'])]:
                    result=dict(seed=seed,regime=regime,model=name,policy=policy,source='DEMO DATA',**metrics(test.y,ts,test.patient,lam))
                    if policy=='crc':result.update({k:crc[k] for k in ['lambda','n','B','alpha','terminal_fallback','loss_definition','assumptions']})
                    summaries.append(result);print('EHR_METRIC',json.dumps(result,default=float),flush=True)
                if seed==cfg['seeds'][0]:
                    intervals.extend(dict(regime=regime,model=name,policy='crc',**r) for r in cluster_bootstrap(z,seed,cfg['bootstrap_replicates']))
                    bins.extend(dict(regime=regime,model=name,**r) for r in calibration_bins(test.y.to_numpy(),ts))
                    for group in ['gender','race']:
                        for value,q in z.groupby(group):
                            m=metrics(q.y,q.score,q.patient,crc['lambda']);subgroups.append(dict(regime=regime,model=name,group=group,value=value,support_status='low_support' if m['n_patients']<20 or m['positive_patients']<5 else 'descriptive',**m))
                    # Frozen model, no recalibration under missingness shift.
                    shifted=test[cols].copy()
                    for col in shifted.columns:
                        if col.startswith(('v2','l5')):shifted[col]=np.nan
                    ps=probability(model,shifted)
                    summaries.append(dict(seed=seed,regime=regime+'_missingness_shift',model=name,policy='crc_without_recalibration',source='DEMO DATA',guarantee='none under shift',**metrics(test.y,ps,test.patient,crc['lambda'])))
            if seed==cfg['seeds'][0]:
                for name in ['metadata','consent_no_labs','consent_no_vitals']:
                    deltas.extend(dict(regime=regime,model_a='full_ehr',model_b=name,**r) for r in paired_bootstrap(paired['full_ehr'],paired[name],seed,cfg['bootstrap_replicates']))
    for name,rows in [('ehr_metrics',summaries),('ehr_intervals',intervals),('ehr_reliability',bins),('ehr_subgroups',subgroups),('ehr_calibration',calibrations),('ehr_paired_effects',deltas)]:pd.DataFrame(rows).to_csv(Path(out,name+'.csv'),index=False)
    Path(out,'ehr_splits.json').write_text(json.dumps(splits,indent=2));pred=pd.concat(predictions,ignore_index=True);pred.to_csv(Path(out,'ehr_predictions.csv'),index=False)
    return pred,pd.DataFrame(summaries)

def provider_graph(root,cohort,out):
    orders=table(root,'hosp','poe',columns=['poe_id','subject_id','hadm_id','ordertime','order_provider_id'])
    orders['ordertime']=pd.to_datetime(orders.ordertime,errors='coerce')
    # One snapshot per patient: first eligible landmark, avoiding future relationship leakage.
    cut=cohort.assign(time=pd.to_datetime(cohort.cutoff)).sort_values('time').groupby('patient').first()
    rows=[];g=nx.DiGraph()
    for patient,x in cut.iterrows():
        q=orders[(orders.subject_id==patient)&(orders.hadm_id==x.hadm_id)&(orders.ordertime<=x.time)].dropna(subset=['order_provider_id'])
        assert (q.ordertime<=x.time).all()
        for a in q.itertuples():
            u='provider:'+str(a.order_provider_id);v='order:'+str(a.poe_id);w='patient:'+str(patient)
            g.add_node(u,type='provider');g.add_node(v,type='order');g.add_node(w,type='patient')
            g.add_edge(u,v,type='authored_order',timestamp=str(a.ordertime));g.add_edge(v,w,type='order_for_patient',timestamp=str(a.ordertime))
        rows.append(dict(patient=patient,orders=len(q),providers=q.order_provider_id.nunique(),cutoff=str(x.time),source='DEMO DATA'))
    # Provider-projection sensitivity; co-ordering does not establish communication.
    bg=nx.Graph();bg.add_edges_from((u,v) for u,v in g.edges)
    providers=[n for n,d in g.nodes(data=True) if d['type']=='provider']
    # Explicit provider-patient incidence; do not project across order node paths incorrectly.
    incidence=nx.Graph();incidence.add_nodes_from(providers,bipartite=0)
    for u in providers:
        for order in g.successors(u):
            for patient in g.successors(order):incidence.add_edge(u,patient)
    projected=nx.algorithms.bipartite.weighted_projected_graph(incidence,providers)
    bet=nx.betweenness_centrality(projected,weight=None)
    pd.DataFrame([dict(provider=n,degree=projected.degree(n),betweenness=bet[n]) for n in providers]).to_csv(Path(out,'provider_centrality.csv'),index=False)
    pd.DataFrame(rows).to_csv(Path(out,'provider_snapshots.csv'),index=False)
    summary={'source':'DEMO DATA','nodes':g.number_of_nodes(),'edges':g.number_of_edges(),'providers':len(providers),
             'provider_projection_edges':projected.number_of_edges(),'semantics':'pre-cutoff ordering proxy; not note-access or validated teamwork',
             'prediction_model':'none; structural descriptive graph only'}
    Path(out,'provider_graph_summary.json').write_text(json.dumps(summary,indent=2));print('PROVIDER_GRAPH',json.dumps(summary),flush=True)
    return summary
