import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from consent_ai.sepsis import causal_features,weighted_auc,auc_structure

def test_causal_sepsis_feature_prefix_is_invariant():
    d=pd.DataFrame({'HR':[80,np.nan,200],'Age':[50,50,50],'ICULOS':[1,2,3],'SepsisLabel':[0,0,1]})
    a=causal_features(d.iloc[:2]);b=causal_features(d)
    pd.testing.assert_frame_equal(a,b.iloc[:2])
    assert a.last_HR.iloc[-1]==80 and 'last_SepsisLabel' not in a

def test_cluster_weighted_auc_matches_explicit_patient_resampling():
    y=np.array([0,1,0,1,1,0]);p=np.array([.1,.5,.5,.8,.5,.1]);w=np.array([2,2,0,0,1,1])
    expected=roc_auc_score(np.repeat(y,w),np.repeat(p,w))
    assert abs(weighted_auc(auc_structure(y,p),w)-expected)<1e-12
