"""Approval-free follow-up: public sepsis cohort, estimand choice and review load."""
from pathlib import Path
import concurrent.futures,hashlib,json,threading,zipfile
import numpy as np
import pandas as pd
import requests
from .data import download,sha
from .evaluation import fit_model,probability,metrics,calibration_bins
from .risk import calibrate,patient_losses,ASSUMPTIONS,LOSS,queue

VITAL=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2']
META=['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']

def acquire(root,out):
    root=Path(root)/'sepsis-public';root.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path('configs/sepsis_selection.json').read_text());local=threading.local()
    archive=root/'sepsis-public-sample.zip'
    # Reuse only known members; individual checksums are still validated.
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for item in selection['files']:
                name=item['site']+'/'+Path(item['key']).name;p=root/name
                if not p.exists():p.parent.mkdir(exist_ok=True);p.write_bytes(z.read(name))
    def get(item):
        p=root/item['site']/Path(item['key']).name;p.parent.mkdir(exist_ok=True)
        url='https://physionet-open.s3.amazonaws.com/'+item['key']
        if not p.exists():
            if not hasattr(local,'session'):local.session=requests.Session()
            response=local.session.get(url,timeout=90);response.raise_for_status();p.write_bytes(response.content)
        assert sha(p,'md5')==item['etag'],f'Checksum mismatch: {p.name}'
        return dict(path=f'data/raw/sepsis-public/{item["site"]}/{p.name}',source_url=url,sha256=sha(p),site=item['site'],source='REAL OPEN DATA',license='CC BY 4.0')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:manifest=list(ex.map(get,selection['files']))
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for item in selection['files']:
                name=item['site']+'/'+Path(item['key']).name;z.write(root/name,arcname=name)
    download('https://physionet.org/files/challenge-2019/1.0.0/LICENSE.txt',root/'LICENSE.txt')
    Path(out,'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    return root

def causal_features(raw):
    assert 'SepsisLabel' in raw
    cols=[c for c in raw if c!='SepsisLabel'];x=raw[cols].copy()
    numeric=x.apply(pd.to_numeric,errors='raise');last=numeric.ffill();measure=[c for c in cols if c not in META]
    mean=numeric[measure].expanding(min_periods=1).mean();count=numeric[measure].notna().cumsum()
    return pd.concat([last.add_prefix('last_'),mean.add_prefix('mean_'),count.add_prefix('count_')],axis=1).astype(np.float32)

def cohort(root,out):
    features=[];records=[];audit=[]
    for site in ['A','B']:
        for p in sorted(Path(root,site).glob('*.psv')):
            d=pd.read_csv(p,sep='|');assert d.shape[1]==41 and d.SepsisLabel.isin([0,1]).all()
            x=causal_features(d);features.append(x)
            records.append(pd.DataFrame(dict(patient=p.stem,site=site,row_id=[f'{p.stem}:{i+1}' for i in range(len(d))],hour=np.arange(1,len(d)+1),y=d.SepsisLabel.astype(int),gender=d.Gender,age=d.Age)))
            audit.append(dict(patient=p.stem,site=site,hours=len(d),positive_hours=int(d.SepsisLabel.sum()),positive_patient=int(d.SepsisLabel.max()),missing_fraction=float(d.drop(columns='SepsisLabel').isna().mean().mean())))
    X=pd.concat(features,ignore_index=True);d=pd.concat(records,ignore_index=True);a=pd.DataFrame(audit)
    assert not d.row_id.duplicated().any();assert X.shape[0]==len(d)
    a.to_csv(Path(out,'cohort.csv'),index=False)
    summary=a.groupby('site').agg(patients=('patient','size'),hours=('hours','sum'),positive_patients=('positive_patient','sum'),positive_hours=('positive_hours','sum')).reset_index()
    summary.to_csv(Path(out,'cohort_summary.csv'),index=False);print('SEPSIS_COHORT',summary.to_json(orient='records'),flush=True)
    return X,d

def auc_structure(y,p):
    _,rank=np.unique(p,return_inverse=True)
    return np.asarray(y),rank

def weighted_auc(structure,w):
    y,rank=structure;pos=np.bincount(rank,weights=w*y);neg=np.bincount(rank,weights=w*(1-y))
    if not pos.sum() or not neg.sum():return np.nan
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))

def bootstrap_comparisons(predictions,out,reps=500):
    rows=[];effects=[]
    rng=np.random.default_rng(2026)
    for site in ['A','B']:
        data=predictions[(predictions.seed==11)&(predictions.site==site)]
        models={n:g.sort_values('row_id').reset_index(drop=True) for n,g in data.groupby('model')}
        ref=models['full_ehr'];ids,inv=np.unique(ref.patient,return_inverse=True);n=len(ids)
        for g in models.values():assert g.row_id.equals(ref.row_id)
        structures={name:auc_structure(g.y.to_numpy(),g.score.to_numpy()) for name,g in models.items()}
        estimates={name:[] for name in models}
        for _ in range(reps):
            counts=rng.multinomial(n,np.repeat(1/n,n));w=counts[inv]
            for name,g in models.items():
                estimates[name].append((weighted_auc(structures[name],w),np.average((g.y.to_numpy()-g.score.to_numpy())**2,weights=w)))
        for name,values in estimates.items():
            arr=np.array(values)
            for j,metric in enumerate(['auroc','brier']):
                valid=arr[:,j][np.isfinite(arr[:,j])];lo,hi=np.quantile(valid,[.025,.975]) if len(valid)>1 else [np.nan,np.nan]
                rows.append(dict(site=site,model=name,metric=metric,lower=lo,upper=hi,valid_replicates=len(valid),requested_replicates=reps,unit='patient',seed=11))
                if name!='full_ehr':
                    delta=arr[:,j]-np.array(estimates['full_ehr'])[:,j];delta=delta[np.isfinite(delta)];dl,dh=np.quantile(delta,[.025,.975])
                    a=models[name];b=models['full_ehr'];point=weighted_auc(structures[name],np.ones(len(ref)))-weighted_auc(structures['full_ehr'],np.ones(len(ref))) if j==0 else np.mean((a.y-a.score)**2)-np.mean((b.y-b.score)**2)
                    effects.append(dict(site=site,model_a='full_ehr',model_b=name,metric=metric,delta_b_minus_a=point,lower=dl,upper=dh,valid_replicates=len(delta),seed=11))
    pd.DataFrame(rows).to_csv(Path(out,'intervals.csv'),index=False);pd.DataFrame(effects).to_csv(Path(out,'paired_effects.csv'),index=False)

def run(root,out,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);root=acquire(root,out);X,d=cohort(root,out)
    meta=['last_'+x for x in META];v=[c for c in X if any(c.endswith('_'+x) for x in VITAL)];labs=[c for c in X if c not in meta+v]
    variants={'metadata':meta,'full_ehr':list(X),'consent_no_labs':meta+v,'consent_no_vitals':meta+labs}
    results=[];predictions=[];calibrations=[];splits=[];groups=[];curves=[];capacity=[]
    for seed in cfg['seeds']:
        ids=np.array(sorted(d[d.site=='A'].patient.unique()));np.random.default_rng(seed).shuffle(ids)
        trainids,calids,testids=np.split(ids,[int(.6*len(ids)),int(.8*len(ids))])
        assert not(set(trainids)&set(calids) or set(trainids)&set(testids) or set(calids)&set(testids))
        splits.append(dict(seed=seed,train=trainids.tolist(),calibration=calids.tolist(),internal_test=testids.tolist(),external_site='B'))
        tr=d.patient.isin(trainids);ca=d.patient.isin(calids);te=(d.patient.isin(testids))|(d.site=='B')
        for name,cols in variants.items():
            model=fit_model(X.loc[tr,cols],d.y[tr]);cs=probability(model,X.loc[ca,cols]);ts=probability(model,X.loc[te,cols])
            c=d[ca];z=d[te].copy();z['score']=ts;z['model']=name;z['seed']=seed;predictions.append(z)
            positive_ids=c.groupby('patient').y.max();positive_ids=positive_ids[positive_ids==1].index;pos=c.patient.isin(positive_ids)
            rules={'fixed_0.5':{'lambda':.5},'crc_unconditional':calibrate(c.y,cs,c.patient,cfg['alpha']),
                   'crc_positive_conditional':calibrate(c.y[pos],cs[pos],c.patient[pos],cfg['alpha'])}
            for rule,crc in rules.items():
                if rule!='fixed_0.5':
                    crc.pop('curve');crc['conditioning']='positive patients only' if rule=='crc_positive_conditional' else 'all patients'
                    calibrations.append(dict(seed=seed,model=name,rule=rule,**crc))
                for site,t in z.groupby('site'):
                    met=metrics(t.y,t.score,t.patient,crc['lambda'])
                    neg=t.y==0;met['false_alerts_per_24_observed_hours']=24*met['false_alerts']/len(t)
                    rec=dict(seed=seed,model=name,site=site,rule=rule,source='REAL OPEN DATA / EXPERIMENTAL CONSENT',lambda_value=crc['lambda'],**met)
                    if rule!='fixed_0.5':rec.update(n=crc['n'],B=1,alpha=.05,conditioning=crc['conditioning'],loss_definition=LOSS,assumptions=ASSUMPTIONS,guarantee_domain='conditional positive patients within A' if rule=='crc_positive_conditional' and site=='A' else 'within A only' if site=='A' else 'none asserted: hospital shift')
                    results.append(rec);print('SEPSIS_METRIC',json.dumps(rec),flush=True)
                    if seed==11:
                        for attr in ['gender','age_group']:
                            temp=t.copy();temp['age_group']=np.where(temp.age>=65,'65+','under65')
                            for value,q in temp.groupby(attr):groups.append(dict(model=name,site=site,rule=rule,group=attr,value=value,**metrics(q.y,q.score,q.patient,crc['lambda'])))
                        for rate in [5,15,30,60]:
                            for K in [2,5,10,20,40,60]:capacity.append(dict(model=name,site=site,rule=rule,opportunities_per_hour=rate,K=K,expected_reviews=rate*met['review_fraction'],feasible=rate*met['review_fraction']<=K,patient_missed_loss=met['patient_missed_loss'],positive_patient_missed_loss=met['positive_patient_missed_loss']))
            if seed==11:
                for site,t in z.groupby('site'):curves.extend(dict(model=name,site=site,**r) for r in calibration_bins(t.y.to_numpy(),t.score.to_numpy()))
                # Corrupt the permitted feature measurements while preserving model and threshold.
                shifted=X.loc[te,cols].copy()
                for col in shifted:
                    if col not in meta:shifted[col]=np.nan if not col.startswith('count_') else 0
                ps=probability(model,shifted)
                for site in ['A','B']:
                    take=z.site==site;rule=rules['crc_positive_conditional'];results.append(dict(seed=seed,model=name,site=site,rule='positive_crc_missingness_shift',source='REAL OPEN DATA with simulated missingness',guarantee_domain='none under shift',**metrics(z.y[take],ps[take],z.patient[take],rule['lambda'])))
    p=pd.concat(predictions,ignore_index=True);p.to_csv(out/'predictions.csv.gz',index=False)
    for filename,rows in [('metrics',results),('calibration',calibrations),('subgroups',groups),('reliability',curves),('capacity',capacity)]:pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
    (out/'splits.json').write_text(json.dumps(splits,indent=2));bootstrap_comparisons(p,out,cfg['bootstrap_replicates'])
    return pd.DataFrame(results)
