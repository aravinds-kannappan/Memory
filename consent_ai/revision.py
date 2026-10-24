from pathlib import Path
import os,json,time,pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from . import revision_data as data,revision_models as models,revision_evaluation as evaluation,revision_stress as stress
from .joint_risk import incident_matrices

def weighted_shift(X,items,split,seq,thresholds,cfg,seed,model):
    # Estimated density ratios are a diagnostic, not a finite-sample guarantee.
    a=split['train'];b=split['calibration_B'];idx=np.r_[a,b];x=X[[items[i]['start'] for i in idx]];y=np.r_[np.zeros(len(a)),np.ones(len(b))]
    domain=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=250));domain.fit(x,y)
    cids=split['calibration'];q=domain.predict_proba(X[[items[i]['start'] for i in cids]])[:,1];weights=np.clip(q/(1-q+1e-6)*len(a)/len(b),.05,20.)
    l,w,z,_=incident_matrices(seq['calibration'],thresholds,once=False);tl,tw,tz,_=incident_matrices(seq['test_B'],thresholds,once=False)
    wp=weights[z];risk=(l[z]*wp[:,None]).sum(0)/(wp.sum()+1e-10);upper=(risk*wp.sum()+wp.max())/(wp.sum()+wp.max());ok=upper<=cfg['alpha'];j=int(np.flatnonzero(ok)[-1]) if ok.any() else 0
    return dict(seed=seed,model=model,site='B',policy='repeated_window_weighted_crc_diagnostic',threshold=float(thresholds[j]),test_risk=float(tl[tz,j].mean()),any_review_fraction=float(tw[:,j].mean()),effective_positive_n=float(wp.sum()**2/np.sum(wp**2)),ratio_clip='[.05,20]',formal_guarantee=False,assumption='Estimated ratios; covariate-shift and conditional invariance unverified')

def run(root,out):
    start=time.time();out=Path(out)/'revision';out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path('configs/revision_protocol.json').read_text())
    provenance=dict(run_id=os.environ.get('ORX_RUN_ID'),commit=os.environ.get('ORX_COMMIT'),configuration=cfg)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2));print('REVISION_CONFIGURATION',json.dumps(provenance),flush=True)
    root=data.acquire(root,out);X,y,items,columns=data.load(root,out)
    pd.DataFrame(items).to_csv(out/'cohort.csv',index=False);print('FULL_COHORT',json.dumps({'patients':len(items),'hours':len(y),'sites':dict(pd.Series([p['site'] for p in items]).value_counts().astype(int))},default=int),flush=True)
    metrics=[];curves=[];certificates=[];groups=[];draws=[];consent=[];queues=[];weighted=[]
    resumed=set()
    if 'resume' in cfg:
        import hashlib,shutil
        r=cfg['resume'];source=out.parent/r['directory']
        for name,digest in r['sha256'].items():
            assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
            if name!='provenance.json':shutil.copy2(source/name,out/name)
        metrics=pd.read_csv(source/'metrics.csv').to_dict('records');curves=pd.read_csv(source/'frontiers.csv').to_dict('records');certificates=pd.read_csv(source/'certificates.csv').to_dict('records');groups=pd.read_csv(source/'groups.csv').to_dict('records');draws=pd.read_csv(source/'calibration_draws.csv').to_dict('records');consent=pd.read_csv(source/'consent.csv').to_dict('records');queues=pd.read_csv(source/'queues.csv').to_dict('records');weighted=pd.read_csv(source/'weighted_shift.csv').to_dict('records')
        resumed={(x['seed'],x['model']) for x in r['completed']}
        print('RESUMED_COMPLETED_MODELS',json.dumps({'models':r['completed'],'source_run':r['source_run'],'source_commit':r['source_commit']}),flush=True)

    for seed in cfg['seeds']:
        split=data.split(items,seed);(out/f'splits-{seed}.json').write_text(json.dumps({k:[items[i]['patient'] for i in ids] for k,ids in split.items()},indent=2))
        for name in cfg['models']:
            if (seed,name) in resumed:continue
            print('MODEL_START',seed,name,flush=True)
            est=models.tune_gru(X,y,items,split,seed,out,cfg['max_epochs']) if name=='gru' else models.tune_tabular(X,y,items,split,name,seed,out)
            if name!='gru':
                with (out/f'{name}-{seed}.pkl').open('wb') as f:pickle.dump(est,f)
            seq={}
            for key in ['validation','calibration','calibration_B','test_A','test_B']:
                pred=models.predict(est,X,items,split[key]);seq[key]=data.sequences(items,split[key],y,pred)
                np.savez_compressed(out/f'predictions-{name}-{seed}-{key}.npz',score=pred.astype('float32'),ids=split[key])
            thresholds=np.unique(np.r_[0,np.quantile(np.concatenate([p['score'] for p in seq['validation']]),np.linspace(0,1,cfg['threshold_count'])),1.000001]);(out/f'thresholds-{name}-{seed}.json').write_text(json.dumps(thresholds.tolist()))
            for site in ['A','B']:
                rec=dict(seed=seed,model=name,site=site,**evaluation.discrimination(seq['test_'+site],cfg['bootstrap_replicates'],seed));metrics.append(rec);print('REVISION_METRIC',json.dumps(rec),flush=True)
            r,c,g,d,mats=evaluation.policy_analysis(seq,thresholds,cfg,seed,name,out);curves+=r;certificates+=c;groups+=g;draws+=d
            # Queue scenario policy is chosen on validation utility, never on test outcomes.
            valutility=evaluation.utility_curve(seq['validation'],thresholds,True);j=int(np.argmax(valutility))
            for site in ['A','B']:queues+=stress.queues(seq['test_'+site],float(thresholds[j]),cfg,seed,name,site)
            consent+=stress.consent_experiment(est,X,y,items,split,columns,thresholds,cfg,seed,name)
            weighted.append(weighted_shift(X,items,split,seq,thresholds,cfg,seed,name))
            for filename,rows in [('metrics',metrics),('frontiers',curves),('certificates',certificates),('groups',groups),('calibration_draws',draws),('consent',consent),('queues',queues),('weighted_shift',weighted)]:pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
            print('MODEL_COMPLETE',seed,name,flush=True)
    stress.random_graphs(cfg,out)
    from .revision_reporting import create
    create(out,cfg,provenance)
    print('REVISION_COMPLETE',json.dumps({'seconds':time.time()-start,'patients':len(items),'models':len(metrics),'output':str(out)}),flush=True)
