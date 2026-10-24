import numpy as np
from consent_ai.consent_comparison import restricted_predict,paired_discrimination,policy_vectors,paired_policy

class RecordingModel:
    def predict_proba(self,x):
        self.seen=x.copy()
        return np.tile([.5,.5],(len(x),1))

def test_revocation_clears_values_and_counts_but_preserves_metadata():
    columns=['last_HR','mean_HR','count_HR','last_Lactate','mean_Lactate','count_Lactate','last_Age']
    x=np.ones((16,7));x[:,6]=60;model=RecordingModel()
    items=[dict(patient='p1',site='A',start=0,end=16)]
    _,exposure=restricted_predict(model,x,items,[0],columns,dict(lab_denial=1,vital_denial=0,start=12),11)
    assert np.array_equal(model.seen[:12],x[:12])
    assert np.isnan(model.seen[12:,3:5]).all()
    assert (model.seen[12:,5]==0).all()
    assert np.array_equal(model.seen[:,6],x[:,6])
    assert np.array_equal(model.seen[:,:3],x[:,:3])
    assert exposure['permitted_source_hour_fraction']==.875
    assert np.isfinite(x).all()  # input remains immutable

def test_identical_predictions_have_exact_zero_paired_intervals():
    y=np.array([0,1,0,1,0,1]);s=np.array([.1,.7,.2,.8,.3,.9]);owners=np.repeat(np.arange(3),2)
    r=paired_discrimination(y,s,s,owners,3,30,11)
    assert r['delta_auroc']==r['delta_auroc_lower']==r['delta_auroc_upper']==0
    assert r['delta_brier_lower']==r['delta_brier_upper']==0

def test_paired_policy_keeps_same_patients_and_loss_definition():
    seq=[{'y':np.array([0]*4+[1]*8),'score':np.array([.1]*4+[.8]*8)}, {'y':np.zeros(12),'score':np.zeros(12)}]*10
    v=policy_vectors(seq,.5,True);r=paired_policy(v,v,30,29)
    for metric in ['risk','reviews','utility','false_reviews_per_negative']:
        assert r['delta_'+metric]==r['delta_'+metric+'_lower']==r['delta_'+metric+'_upper']==0
