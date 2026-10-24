"""Joint risk/workload certificates for finite, validation-frozen policy grids.

This is an application of simultaneous binomial confidence bounds / Learn then
Test, not a new distribution-free theorem. Unlike monotone CRC, it supports
non-monotone first-alert incident loss.
"""
import numpy as np
from scipy.stats import beta

def binomial_bounds(k,n,tail):
    k=np.asarray(k);n=np.asarray(n)
    low=np.where(k==0,0,beta.ppf(tail,k,np.maximum(1,n-k+1)))
    high=np.where(k==n,1,beta.ppf(1-tail,k+1,np.maximum(1,n-k)))
    return np.where(n>0,low,0.),np.where(n>0,high,1.)

def certify(loss,review,positive,alpha=.05,budget=.25,delta=.05):
    """Matrices patient x threshold. Positive specifies incident-eligible events.
    Return simultaneous intervals, certified options and resolved infeasibility.
    Empty certified set alone means insufficient evidence, not infeasibility.
    """
    loss=np.asarray(loss,bool);review=np.asarray(review,bool);pos=np.asarray(positive,bool)
    assert loss.shape==review.shape and loss.ndim==2
    m=loss.shape[1];tail=delta/(4*m)
    rl,ru=binomial_bounds(loss[pos].sum(0),pos.sum(),tail)
    wl,wu=binomial_bounds(review.sum(0),len(review),tail)
    feasible=(ru<=alpha)&(wu<=budget)
    impossible=(rl>alpha)|(wl>budget)
    choice=int(np.flatnonzero(feasible)[np.argmin(wu[feasible])]) if feasible.any() else None
    status='certified' if feasible.any() else 'grid_infeasible' if impossible.all() else 'undetermined'
    return dict(risk_lower=rl,risk_upper=ru,workload_lower=wl,workload_upper=wu,certified=feasible,choice=choice,status=status,n_positive=int(pos.sum()),n=len(pos),delta=delta,alpha=alpha,budget=budget)

def incident_matrices(sequences,thresholds,once=True):
    thresholds=np.asarray(thresholds);loss=[];reviews=[];first=[];positive=[]
    for p in sequences:
        score=np.asarray(p['score']);y=np.asarray(p['y']);where=np.flatnonzero(y)
        onset=int(where[0]+6) if len(where) else len(y)
        eligible=bool(len(where) and where[0]>0 and onset<len(y))
        stop=min(onset,len(y));cross=score[:stop,None]>=thresholds[None,:]
        exists=cross.any(0);times=np.where(exists,cross.argmax(0),-1) if stop else np.full(len(thresholds),-1)
        success=(times>=int(where[0]))&exists if eligible else np.zeros(len(thresholds),bool)
        if eligible and not once:success=cross[int(where[0]):stop].any(0)
        loss.append(~success);reviews.append(exists);first.append(times);positive.append(eligible)
    return np.asarray(loss),np.asarray(reviews),np.asarray(positive),np.asarray(first)

def monotone_crc(loss,positive,alpha=.05):
    """Thresholds must run most permissive to strict; reject non-monotone loss."""
    a=np.asarray(loss)[np.asarray(positive)]
    if a.shape[0]==0:return None
    if np.any(np.diff(a.astype(int),axis=1)<0):raise ValueError('Loss is not monotone: ordinary CRC is invalid for this policy.')
    ok=(a.sum(0)+1)/(len(a)+1)<=alpha
    return int(np.flatnonzero(ok)[-1]) if ok.any() else 0
