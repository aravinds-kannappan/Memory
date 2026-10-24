"""Equations 4–5. Calibration units are entire patients, never frames."""
import numpy as np

ASSUMPTIONS = ('Frozen predictor and loss; exchangeable calibration/future patient loss functions; '
               'bounded B=1, non-increasing right-continuous loss in lambda; terminal all-alert loss=0; '
               'no post-calibration alert suppression. Expectation over calibration and new patient, '
               'not a per-patient/subgroup guarantee. Shifted populations have no asserted guarantee.')
LOSS = 'L_i = missed positive windows / max(1, number of positive windows); zero for no positive windows'

def patient_losses(y, scores, patients, lam):
    y, scores, patients = np.asarray(y), np.asarray(scores), np.asarray(patients)
    if not (len(y) == len(scores) == len(patients)) or not len(y):
        raise ValueError('Nonempty aligned observations required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Binary outcomes and finite [0,1] scores required')
    if not 0 <= lam <= 1: raise ValueError('lambda outside [0,1]')
    # Comparing lambda to 1-score preserves exact event knots and right-continuity.
    alert = (1 - scores) <= lam
    return np.array([np.sum(y[patients==p] * ~alert[patients==p]) /
                     max(1, np.sum(y[patients==p])) for p in np.unique(patients)])

def calibrate(y, scores, patients, alpha=0.05):
    if not 0 < alpha < 1: raise ValueError('alpha outside (0,1)')
    n = len(np.unique(patients)); B=1.0
    knots = np.unique(np.r_[0., 1-np.asarray(scores), 1.])
    # Patient-weighted positive rows yield exactly the mean patient loss,
    # with tied knots evaluated after all tied alerts enter the nested set.
    y=np.asarray(y);scores=np.asarray(scores);patients=np.asarray(patients)
    patient_losses(y,scores,patients,0.)  # validate before cumulative evaluation
    weights=np.zeros(len(y))
    for unit in np.unique(patients):
        keep=patients==unit;weights[keep]=y[keep]/max(1,y[keep].sum())/n
    order=np.argsort(1-scores);events=(1-scores)[order];cumulative=np.r_[0.,np.cumsum(weights[order])]
    risks=np.maximum(0.,weights.sum()-cumulative[np.searchsorted(events,knots,side='right')])
    curve=[]; selected=None
    for lam,risk in zip(knots,risks):
        bound=(n*risk+B)/(n+1)
        curve.append({'lambda':float(lam),'empirical_risk':float(risk),'corrected_risk':float(bound)})
        if selected is None and bound<=alpha: selected=float(lam)
    fallback=selected is None
    if fallback: selected=1.0
    return {'lambda':selected,'threshold':1-selected,'n':n,'B':B,'alpha':alpha,
            'terminal_fallback':fallback,'calibration_risk':float(patient_losses(y,scores,patients,selected).mean()),
            'loss_definition':LOSS,'assumptions':ASSUMPTIONS,'curve':curve}

def expected_capacity(probabilities, K):
    p=np.asarray(probabilities,dtype=float)
    if K<0 or not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid capacity/probability')
    load=float(p.sum())
    return {'expected_reviews':load,'K':float(K),'feasible':bool(load<=K)}

def queue(arrival_minutes, service_minutes, period_minutes=480, queue_limit=20):
    """One FCFS server. Overflow is counted, never silently folded into CRC."""
    if service_minutes<=0: raise ValueError('Positive service time required')
    finish=[]; waits=[]; dropped=0; peak=0
    for arrival in sorted(arrival_minutes):
        active=sum(t>arrival for t in finish)
        peak=max(peak,active)
        if active>=queue_limit+1: dropped+=1; continue
        start=max(arrival,finish[-1] if finish else arrival)
        waits.append(start-arrival); finish.append(start+service_minutes)
    return {'arrivals':len(arrival_minutes),'accepted':len(finish),'overflow':dropped,
            'completed_by_period':sum(t<=period_minutes for t in finish),
            'backlog_at_end':sum(t>period_minutes for t in finish),
            'max_queue_including_service':peak,
            'mean_wait_minutes':float(np.mean(waits)) if waits else 0.,
            'p95_wait_minutes':float(np.quantile(waits,.95)) if waits else 0.}
