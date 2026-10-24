import numpy as np
from consent_ai.joint_risk import incident_matrices,certify,monotone_crc,binomial_bounds

def test_first_alarm_can_break_monotonicity():
    # Onset hour 8; timely window is hours 2..7. Lower threshold fires too early.
    p={'score':[.6,0,0,0,.9,0,0,0,0,0], 'y':[0,0,1,1,1,1,1,1,1,1]}
    loss,review,pos,t=incident_matrices([p],[.5,.8,1.])
    assert pos[0] and loss.tolist()==[[True,False,True]] and t.tolist()==[[0,4,-1]]
    import pytest
    with pytest.raises(ValueError):monotone_crc(loss,pos)
    repeated=incident_matrices([p],[.5,.8,1.],once=False)[0]
    assert repeated.tolist()==[[False,False,True]]

def test_certificate_is_not_empirical_or_unconditional_risk():
    loss=np.zeros((1000,2),bool);review=np.zeros_like(loss);positive=np.arange(1000)<10
    c=certify(loss,review,positive)
    assert c['status']=='undetermined' and c['risk_upper'][0]>.05
    positive[:]=True;c=certify(loss,review,positive)
    assert c['status']=='certified'
    c=certify(np.ones_like(loss),review,positive)
    assert c['status']=='grid_infeasible'

def test_exact_binomial_edges_and_no_data():
    lo,hi=binomial_bounds(np.array([0,10,0]),np.array([10,10,0]),.025)
    assert lo[0]==0 and hi[1]==1 and lo[2]==0 and hi[2]==1
    assert np.isclose(hi[0],1-.025**.1)

def test_official_utility_vectorization():
    from consent_ai.revision_evaluation import utility_curve
    from consent_ai.vendor.evaluate_sepsis_score import compute_prediction_utility
    y=np.r_[np.zeros(5),np.ones(15)];scores=np.linspace(0,1,20);p={'y':y,'score':scores}
    out=utility_curve([p],[.25,.75],False)
    best=np.zeros(20);best[:15]=1
    base=compute_prediction_utility(y,np.zeros(20));denom=compute_prediction_utility(y,best)-base
    expected=[(compute_prediction_utility(y,scores>=t)-base)/denom for t in [.25,.75]]
    assert np.allclose(out,expected)

def test_boosting_handles_constant_and_empty_features():
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    x=np.column_stack([np.arange(40),np.zeros(40),np.full(40,np.nan)])
    estimator=make_pipeline(VarianceThreshold(),HistGradientBoostingClassifier(max_iter=2,min_samples_leaf=2))
    estimator.fit(x,np.arange(40)%2)
    assert estimator.predict_proba(x).shape==(40,2)
