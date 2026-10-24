import itertools
import networkx as nx
import numpy as np
import pytest
from consent_ai.graph import *
from consent_ai.risk import *

def test_masks_patient_purpose_and_storage():
    g=pipeline(bypass=True);p={('p0',s,'fall_detection'):True for s in ['video','ehr']}
    h,A,M=mask(g,'p0','fall_detection',p)
    assert np.array_equal(nx.to_numpy_array(h,nodelist=list(g)),A*M)
    assert nx.has_path(h,'video','pager')
    for sink in ['archive','profile','ehr_write']:assert not nx.has_path(h,'video',sink)
    assert mask(g,'p1','fall_detection',p)[0].number_of_edges()==0
    assert mask(g,'p0','profiling',p)[0].number_of_edges()==0

def test_review_equivalence_exhaustive_dags():
    # Exhaustively compare node-deletion criterion to all original paths.
    nodes=['s','a','h','d'];possible=list(itertools.combinations(nodes,2))
    for bits in itertools.product([0,1],repeat=len(possible)):
        g=nx.DiGraph();g.add_nodes_from(nodes);g.add_edges_from(e for e,b in zip(possible,bits) if b)
        expected=[p for p in nx.all_simple_paths(g,'s','d') if 'h' not in p]
        assert violating_paths(g,['s'],['d'],['h'])==expected

def test_bypass_and_vacuity():
    assert violating_paths(pipeline(True),['video'],['action'],['physician'])
    g=pipeline();assert not violating_paths(g,['video'],['action'],['physician'])
    g.remove_node('summary');assert not nx.has_path(g,'video','action')

def test_temporal_order():
    g=nx.DiGraph();g.add_edge('s','a',timestamp=5,event_only=True,latency=0)
    g.add_edge('a','d',timestamp=3,event_only=True,latency=0)
    assert nx.has_path(g,'s','d') and temporal_path(g,'s','d')[0] is None

def test_withdrawal_cached_derivatives():
    g=pipeline();p={('p0',s,'archiving'):5. for s in ['video','ehr']}
    assert nx.has_path(mask(g,'p0','archiving',p,time=4)[0],'cache','archive')
    assert not nx.has_path(mask(g,'p0','archiving',p,time=5)[0],'cache','archive')
    a=[{'id':n,'patient':'p0','provenance':['video']} for n in ['raw','feature','summary','new_derivative']]
    assert len(revoke_artifacts(a,'p0','video','archiving',5))==4
    assert all(not usable(x,'archiving',5) and usable(x,'fall_detection',5) for x in a)

def test_crc_finite_sample_ties_monotonic():
    y=np.ones(40);s=np.repeat([.2,.8],20);p=np.arange(40)
    r=calibrate(y,s,p,.05)
    assert r['lambda']==.8 and r['n']==40 and not r['terminal_fallback']
    losses=np.array([patient_losses(y,s,p,l) for l in np.linspace(0,1,101)])
    assert np.all(np.diff(losses,axis=0)<=0) and losses.min()>=0 and losses.max()<=1
    assert patient_losses(y,s,p,1).sum()==0

def test_crc_small_n_terminal_not_fake_sample_size():
    r=calibrate([1,1,0],[.2,.8,.1],['a','a','b'],.05)
    assert r['n']==2 and r['terminal_fallback'] and r['lambda']==1

def test_capacity_and_queue_are_different():
    assert expected_capacity([.2,.8],1)['feasible']
    assert not expected_capacity([.2,.8],.9)['feasible']
    q=queue([0,0,0],10,period_minutes=5,queue_limit=1)
    assert q['overflow']==1 and q['backlog_at_end']==2 and q['completed_by_period']==0

def test_minimum_cut_endpoints():
    g=nx.DiGraph([('s','a'),('a','d'),('s','b'),('b','d')])
    assert set(protected_vertex_cut(g,'s','d'))=={'a','b'}
    g.add_edge('s','d');assert protected_vertex_cut(g,'s','d') is None

def test_withdrawal_of_either_parent_blocks_merged_derivative():
    g=pipeline()
    for source in ['video','ehr']:
        p={('p0',s,'archiving'):True for s in ['video','ehr']};p[('p0',source,'archiving')]=False
        h=mask(g,'p0','archiving',p)[0]
        assert not nx.has_path(h,'cache','archive')

def test_vectorized_losses_match_direct_patient_definition():
    rng=np.random.default_rng(2026);u=np.repeat(np.arange(10),7);y=rng.integers(0,2,len(u));p=rng.random(len(u))
    expected=[np.sum(y[u==i]*((1-p[u==i])>.3))/max(1,sum(y[u==i])) for i in range(10)]
    np.testing.assert_allclose(patient_losses(y,p,u,.3),expected)
