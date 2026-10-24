"""Structural interventions, workload sweeps and labeled synthetic CRC validation."""
from pathlib import Path
import json
import networkx as nx
import numpy as np
import pandas as pd
from .graph import pipeline,mask,violating_paths,temporal_path,protected_vertex_cut,revoke_artifacts,usable
from .risk import calibrate,patient_losses,expected_capacity,queue,ASSUMPTIONS,LOSS

def structural(out):
    permissions={('p0',s,p):True for s in ['video','ehr'] for p in ['fall_detection','archiving','profiling']}
    original=pipeline(bypass=True);reviewed=pipeline(bypass=False)
    scenarios={'permissive_bypass':original,'mandatory_review':reviewed,
               'consent_and_review':mask(reviewed,'p0','fall_detection',permissions)[0]}
    rows=[];central=[];stresses=[];witnesses={}
    for name,g in scenarios.items():
        bad=violating_paths(g,['video','ehr'],['action'],['physician'])
        storage=sum(nx.has_path(g,'video',d) for d in ['archive','profile','ehr_write'])
        rows.append(dict(scenario=name,source='SIMULATED SYSTEM GRAPH',nodes=len(g),edges=g.number_of_edges(),unreviewed_paths=len(bad),reachable_storage_sinks=storage,
                         nurse_alert_reachable=nx.has_path(g,'video','pager'),action_reachable=nx.has_path(g,'video','action'),minimum_vertex_cut=json.dumps(protected_vertex_cut(g,'video','action'))))
        witnesses[name]=bad
        bc=nx.betweenness_centrality(g)
        for n in g:
            central.append(dict(scenario=name,node=n,node_type=g.nodes[n]['type'],degree=g.degree(n),in_degree=g.in_degree(n),out_degree=g.out_degree(n),betweenness=bc[n]))
            for intervention in ['remove','delay','corrupt']:
                h=g.copy()
                if intervention=='remove':h.remove_node(n)
                if intervention=='delay':
                    for _,v in list(h.out_edges(n)):h[n][v]['latency']+=20
                reachable=lambda d: 'video' in h and d in h and nx.has_path(h,'video',d)
                timely=temporal_path(h,'video','action',deadline=15)[0] is not None if 'video' in h and 'action' in h else False
                downstream=nx.descendants(g,n) if intervention=='corrupt' else set()
                stresses.append(dict(scenario=name,node=n,intervention=intervention,prediction_reachable=reachable('prediction'),review_reachable=reachable('physician'),action_reachable=reachable('action'),action_within_15_seconds=timely,
                                     exposed_components=len(downstream),action_exposed='action' in downstream,
                                     unreviewed_paths=len(violating_paths(h,['video','ehr'],['action'],['physician']))))
    # Withdrawal: direct source-edge deletion cannot disable already-cached summaries.
    rev=[]
    for provenance in [False,True]:
        p={('p0',s,'archiving'):4. for s in ['video','ehr']}
        for t in [3.,4.,5.]:
            g=mask(reviewed,'p0','archiving',p,time=t,provenance=provenance)[0]
            rev.append(dict(policy='provenance_aware' if provenance else 'direct_source_only',time=t,
                            raw_to_archive=nx.has_path(g,'video','archive'),cached_to_archive=nx.has_path(g,'cache','archive'),
                            summary_to_profile=nx.has_path(g,'summary','profile')))
    for filename,data in [('graph_scenarios',rows),('graph_centrality',central),('graph_stress',stresses),('withdrawal',rev)]:pd.DataFrame(data).to_csv(Path(out,filename+'.csv'),index=False)
    Path(out,'bypass_witnesses.json').write_text(json.dumps(witnesses,indent=2))
    Path(out,'implemented_graph.json').write_text(json.dumps(nx.node_link_data(scenarios['consent_and_review']),indent=2))
    print('GRAPH_SCENARIOS',json.dumps(rows),flush=True)
    return scenarios,pd.DataFrame(rows)

def workload(predictions,cfg,out):
    # Each observed decision opportunity is re-used only as a score distribution,
    # never as extra independent patients for calibration.
    base=predictions[(predictions.model=='full_ehr')&(predictions.seed==cfg['seeds'][0])&(predictions.regime=='random')]
    rows=[];queues=[]
    for lam in np.unique(np.r_[cfg['simulation']['lambda_grid'],base['lambda'].iloc[0]]):
        fraction=float(np.mean((1-base.score.to_numpy())<=lam))
        loss=float(patient_losses(base.y,base.score,base.patient,float(lam)).mean())
        for arrival_rate in cfg['simulation']['arrivals_per_hour']:
            for K in cfg['simulation']['K_per_hour']:
                cap=expected_capacity(np.repeat(fraction,arrival_rate),K)
                rows.append(dict(lambda_value=lam,review_fraction=fraction,patient_missed_loss=loss,opportunities_per_hour=arrival_rate,
                                 source='DEMO scores under SIMULATED arrival rates',**cap))
            for service in cfg['simulation']['service_minutes']:
                for seed in cfg['seeds']:
                    rng=np.random.default_rng(seed);duration=cfg['simulation']['hours']*60
                    times=np.cumsum(rng.exponential(60/arrival_rate,size=int(arrival_rate*cfg['simulation']['hours']*3+100)))
                    times=times[times<=duration];scores=rng.choice(base.score.to_numpy(),len(times));alerts=times[(1-scores)<=lam]
                    q=queue(alerts,service,duration,cfg['simulation']['queue_limit'])
                    queues.append(dict(lambda_value=lam,arrival_rate=arrival_rate,service_minutes=service,seed=seed,source='SIMULATED QUEUE / DEMO SCORE DISTRIBUTION',**q))
    pd.DataFrame(rows).to_csv(Path(out,'capacity_sweep.csv'),index=False);pd.DataFrame(queues).to_csv(Path(out,'queue_sweep.csv'),index=False)
    print('WORKLOAD_SWEEP',json.dumps({'capacity_rows':len(rows),'queue_rows':len(queues),'source':'SIMULATED QUEUES'}),flush=True)
    return pd.DataFrame(rows),pd.DataFrame(queues)

def synthetic_patients(rng,n,shift=False):
    # Independent patients; correlated 8-window trajectories within each patient.
    latent=rng.normal(size=(n,1));u=rng.uniform(size=(n,1))
    y=(rng.uniform(size=(n,8))<(.08+.35*(u<.35))).astype(int)
    logits=-2+2.8*y+.7*latent+rng.normal(size=(n,8))
    if shift:logits-=2*y
    scores=1/(1+np.exp(-logits))
    return y.ravel(),scores.ravel(),np.repeat(np.arange(n),8)

def monte_carlo(cfg,out):
    rows=[]
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed)
        for n in [10,25,100]:
            for rep in range(200):
                y,p,u=synthetic_patients(rng,n);c=calibrate(y,p,u,cfg['alpha'])
                for shifted in [False,True]:
                    ty,tp,tu=synthetic_patients(rng,100,shifted)
                    rows.append(dict(seed=seed,replicate=rep,n=n,B=1,alpha=cfg['alpha'],lambda_value=c['lambda'],terminal_fallback=c['terminal_fallback'],
                                     shifted=shifted,risk=patient_losses(ty,tp,tu,c['lambda']).mean(),review_fraction=np.mean(1-tp<=c['lambda']),
                                     source='SYNTHETIC MATHEMATICAL CHECK',loss_definition=LOSS,assumptions=ASSUMPTIONS))
    d=pd.DataFrame(rows);d.to_csv(Path(out,'crc_monte_carlo.csv'),index=False)
    summary=[]
    for (n,shift),x in d.groupby(['n','shifted']):
        sem=x.risk.std(ddof=1)/np.sqrt(len(x));summary.append(dict(n=int(n),B=1,alpha=cfg['alpha'],shifted=bool(shift),replicates=len(x),mean_risk=x.risk.mean(),mc_lower=x.risk.mean()-1.96*sem,mc_upper=x.risk.mean()+1.96*sem,review_fraction=x.review_fraction.mean(),fallback_fraction=x.terminal_fallback.mean(),source='SYNTHETIC',loss_definition=LOSS,assumptions=ASSUMPTIONS))
    pd.DataFrame(summary).to_csv(Path(out,'crc_monte_carlo_summary.csv'),index=False)
    print('CRC_MONTE_CARLO',json.dumps(summary),flush=True)
    return pd.DataFrame(summary)

def integrated_pairing(motion,ehr,cfg,out):
    """Match source participants one-to-one within development roles.

    The label remains the observed staged impact; EHR age/sedative status are
    attached profiles, not evidence that sedation caused an event.
    """
    rows=[]
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed)
        for heldout in sorted(motion.patient.unique()):
            m=motion[(motion.seed==seed)&(motion.heldout==heldout)&(motion.model=='stgcn')]
            profiles=ehr[(ehr.seed==seed)&(ehr.regime=='random')&(ehr.model=='full_ehr')]
            ids=np.array(sorted(profiles.patient.unique()));rng.shuffle(ids)
            mapping=dict(zip(sorted(motion.patient.unique()),ids[:motion.patient.nunique()]))
            for x in m.itertuples():
                profile=profiles[profiles.patient==mapping[x.patient]].iloc[0]
                rows.append(dict(seed=seed,motion_participant=x.patient,ehr_patient=int(profile.patient),motion_row=x.row_id,
                                 ehr_row=profile.row_id,impact_label=x.y,motion_score=x.score,ehr_mortality_score=profile.score,
                                 source='SEMI-SYNTHETIC: open skeleton + DEMO EHR',independent_unit='motion participant, not paired windows'))
    pd.DataFrame(rows).to_csv(Path(out,'semi_synthetic_pairs.csv'),index=False)
    print('INTEGRATED_PAIRING',json.dumps({'pairs':len(rows),'source':'SEMI-SYNTHETIC','risk_interpretation':'No clinical association inferred; no new independent patients created'}),flush=True)
