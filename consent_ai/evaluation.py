import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from .risk import patient_losses, LOSS, ASSUMPTIONS

def fit_model(X,y):
    estimator=LogisticRegression(C=1,max_iter=2000,solver='lbfgs') if len(np.unique(y))==2 else DummyClassifier(strategy='prior')
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),estimator)
    return model.fit(X,y)

def probability(model,X):
    p=model.predict_proba(X);classes=model.classes_
    return p[:,list(classes).index(1)] if 1 in classes else np.zeros(len(X))

def metrics(y,p,patients,lam=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p);patients=np.asarray(patients)
    alert=(1-p)<=lam;positive=y==1;negative=~positive
    loss=patient_losses(y,p,patients,lam)
    positive_units=[u for u in np.unique(patients) if y[patients==u].sum()>0]
    posloss=[patient_losses(y[patients==u],p[patients==u],patients[patients==u],lam)[0] for u in positive_units]
    detected=[bool(np.any(alert[patients==u] & positive[patients==u])) for u in positive_units]
    return dict(n_rows=len(y),n_patients=len(np.unique(patients)),positive_rows=int(y.sum()),positive_patients=len(positive_units),
                auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                average_precision=float(average_precision_score(y,p)) if y.sum() else np.nan,
                brier=float(brier_score_loss(y,p)),
                sensitivity=float(alert[positive].mean()) if positive.any() else np.nan,
                false_positive_rate=float(alert[negative].mean()) if negative.any() else np.nan,
                false_alerts=int(np.sum(alert&negative)),review_fraction=float(alert.mean()),
                patient_missed_loss=float(loss.mean()),positive_patient_missed_loss=float(np.mean(posloss)) if posloss else np.nan,
                positive_patient_detection=float(np.mean(detected)) if detected else np.nan)

def cluster_bootstrap(df, seed=11, reps=500):
    rng=np.random.default_rng(seed);units=df.patient.unique();vals=[]
    for _ in range(reps):
        draws=rng.choice(units,len(units),replace=True)
        chunks=[]
        for j,u in enumerate(draws):
            z=df[df.patient==u].copy();z['patient']=j;chunks.append(z)
        b=pd.concat(chunks,ignore_index=True);vals.append(metrics(b.y,b.score,b.patient,float(b['lambda'].iloc[0])))
    out=[]
    for key in ['auroc','average_precision','brier','sensitivity','false_positive_rate','review_fraction','patient_missed_loss']:
        x=np.array([v[key] for v in vals]);x=x[np.isfinite(x)]
        lo,hi=np.quantile(x,[.025,.975]) if len(x)>1 else [np.nan,np.nan]
        out.append({'metric':key,'lower':lo,'upper':hi,'valid_replicates':len(x),'requested_replicates':reps,'unit':'patient','interval':'percentile 95%'})
    return out

def paired_bootstrap(a,b,seed=11,reps=500):
    cols=['patient','row_id','y','score'];d=a[cols].merge(b[cols],on=['patient','row_id','y'],suffixes=('_a','_b'),validate='one_to_one')
    rng=np.random.default_rng(seed);units=d.patient.unique();out=[]
    if not len(units):return out
    for key,fun in [('brier',brier_score_loss),('auroc',roc_auc_score)]:
        deltas=[]
        for _ in range(reps):
            x=pd.concat([d[d.patient==u] for u in rng.choice(units,len(units),replace=True)])
            if key=='auroc' and x.y.nunique()<2:continue
            deltas.append(fun(x.y,x.score_b)-fun(x.y,x.score_a))
        estimate=fun(d.y,d.score_b)-fun(d.y,d.score_a) if key!='auroc' or d.y.nunique()==2 else np.nan
        lo,hi=np.quantile(deltas,[.025,.975]) if len(deltas)>1 else [np.nan,np.nan]
        out.append(dict(metric=key,delta_b_minus_a=estimate,lower=lo,upper=hi,valid_replicates=len(deltas)))
    return out

def calibration_bins(y,p):
    out=[]
    for lo in np.arange(0,1,.1):
        mask=(p>=lo)&(p<lo+.1 if lo<.9 else p<=1)
        out.append({'lower_edge':lo,'n':int(mask.sum()),'mean_prediction':float(p[mask].mean()) if mask.any() else np.nan,'event_fraction':float(y[mask].mean()) if mask.any() else np.nan})
    return out
