from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .orx_figstyle import use_style,save,PALETTE,TEXT

def create(out,cfg,provenance):
    out=Path(out);m=pd.read_csv(out/'metrics.csv');f=pd.read_csv(out/'frontiers.csv');c=pd.read_csv(out/'certificates.csv');cohort=pd.read_csv(out/'cohort.csv');fd=out/'figures';fd.mkdir(exist_ok=True);use_style()
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),sharey=True)
    for ax,site in zip(axes,['A','B']):
        for model,color in zip(cfg['models'],[PALETTE['blue'],PALETTE['orange'],PALETTE['green']]):
            d=f[(f.site==site)&(f.seed==11)&(f.model==model)&(f.policy=='once')].sort_values(['reviews_per_patient','conditional_incident_miss']);keep=[];best=2
            for _,r in d.iterrows():
                if r.conditional_incident_miss<best:keep.append(r);best=r.conditional_incident_miss
            d=pd.DataFrame(keep);ax.plot(d.reviews_per_patient,d.conditional_incident_miss,'o-',color=color,markersize=3,label=model)
        ax.axhline(.05,color='#555',linestyle='--',linewidth=.8)
        for k in [.25,.5,.75]:ax.axvline(k,color='#bbb',linestyle=':',linewidth=.6)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel(f'Hospital {site}: reviews per patient');ax.legend(fontsize=7,frameon=False)
    axes[0].set_ylabel('Conditional incident missed-event risk');fig.subplots_adjust(left=.09,right=.98,bottom=.2,top=.96,wspace=.08);save(fig,str(fd/'incident_risk_workload'),formats=('pdf','svg','png'))
    shutil.copy('consent_ai/revision_reporting.py',fd/'revision_reporting_source.py');shutil.copy('consent_ai/orx_figstyle.py',fd/'orx_figstyle.py')
    # Every row separates empirical benchmark criteria from a statistical certificate.
    table=[]
    for _,r in f.iterrows():
        table.append(dict(seed=r.seed,model=r.model,policy=r.policy,loss='conditional incident missed event',site=r.site,threshold=r.threshold,useful_benchmark=r.challenge_utility>0,risk_under_alpha=r.conditional_incident_miss<=.05,workload_under_K=r.reviews_per_patient<=.5,K=.5,storage_severed='modeled graph only',review_enforced='structural only',clinical_oversight_measured=False))
    pd.DataFrame(table).to_csv(out/'feasibility.csv',index=False)
    draws=pd.read_csv(out/'calibration_draws.csv');summary=draws.groupby(['seed','model','site','n_positive']).agg(draws=('draw','size'),certification_fraction=('certified','mean'),fraction_exceeding_alpha_among_certified=('exceeds_alpha','mean'),mean_test_risk=('test_risk','mean')).reset_index();summary.to_csv(out/'calibration_draw_summary.csv',index=False)
    stats=dict(status='complete',patients=len(cohort),seed_count=len(cfg['seeds']),models=cfg['models'],mean_metrics=m.groupby(['model','site'])[['auroc','average_precision','brier']].mean().reset_index().to_dict('records'),certificate_counts=c.groupby(['site','calibration_site','status']).size().reset_index(name='count').to_dict('records'),run_id=provenance['run_id'],commit=provenance['commit'])
    (out/'summary.json').write_text(json.dumps(stats,indent=2));print('REVISION_SUMMARY',json.dumps(stats),flush=True)
    lines=['# Full-release incident-alert revision','',f"Completed on {len(cohort):,} public patients. Run `{provenance['run_id']}`, commit `{provenance['commit']}`.",'',
    '## Design','All public A/B files are included. A is split 55/15/15/15% into train, validation, calibration and test; B has a separate 25% calibration reserve and 75% frozen test. Three seeds; validation tunes logistic regression, gradient boosting and a GRU. Training and primary AUROC exclude post-onset rows. Complete challenge utility is reported separately using the official evaluation implementation.',
    'The incident endpoint requires the first alarm in [first positive label, inferred onset). Inferred onset is first positive label plus six hours. Initial-positive and onset-after-discharge patients are excluded from conditional incident risk, retained for review workload. This is an observable benchmark endpoint, not verified clinical onset. Patient-specific opt-out and hour-12 revocation are simulated; recurrent state is cleared at withdrawal.',
    '', '## Main measurements']
    for r in stats['mean_metrics']:lines.append(f"- {r['model']}, hospital {r['site']}: mean pre-onset AUROC {r['auroc']:.4f}, AP {r['average_precision']:.4f}, Brier {r['brier']:.4f}.")
    lines+=['','![Incident risk versus reviews per patient](figures/incident_risk_workload.svg)','The plotted seed-11 test Pareto frontier is descriptive, not a threshold-selection procedure. All candidate thresholds are frozen using validation data. The horizontal line is alpha=.05; vertical references are K=.25/.5/.75. Full seed-specific curves are in `frontiers.csv`.','',
    '## What is certified','`certificates.csv` reports joint upper bounds for conditional incident risk and review probability on a fixed threshold grid. Two-sided exact binomial intervals use tail delta/(4M), giving simultaneous coverage at least 1-delta=.95 under independent patient sampling within the target population. There is no monotonicity assumption. A policy is certified only when both upper bounds meet alpha=.05 and K. An empty certified set is UNDETERMINED unless every grid policy has a lower bound exceeding at least one target. That certificate rules out only this fixed grid, not every possible model or policy. A-to-B certificates have no asserted transfer guarantee.',
    'Repeated-alert incident loss remains monotone; first-alert loss need not be. Lowering a threshold can consume the one permitted alert too early. The tests include a concrete counterexample. Ordinary monotone CRC must not be applied to that first-alert loss. This finite-grid certificate is an application of Learn then Test / simultaneous confidence bounds, not a claimed invention of joint risk control.',
    '', '## Consent, site transfer and oversight','`consent.csv` gives opt-out and revocation cost curves. `groups.csv` allocates delta across four prespecified overlapping demographic groups. `calibration_draw_summary.csv` describes 200 repeated subsets for each available positive-calibration size; draws overlap and are not independent clinical experiments. `weighted_shift.csv` is an exploratory estimated-ratio weighted-CRC comparison for the monotone repeated-window policy, with clipped estimated ratios and no claimed finite-sample guarantee.',
    '`queues.csv` runs arrival/service/noisy-review/fatigue scenarios on the sepsis scores. Service and arrival rates are sensitivity parameters, not measured ward constants. End-to-end misses count timely completed true-positive reviews; false actions and waiting are separate. `graph_ensemble.csv` covers 500 random DAGs with 50–300 nodes; these broaden structural tests but remain simulated.',
    '', '## Answer and boundaries','`feasibility.csv` makes the criteria explicit: positive challenge utility is a benchmark usefulness criterion; empirical risk and workload are separate from their confidence certificates. Consent/storage and review-path checks concern the modeled graph. Even an all-positive benchmark row would not establish actual clinician effectiveness or patient benefit. The theorem and counterexample sharpen what can be guaranteed; the data determine whether this policy/model family meets the declared targets.',
    'Full MIMIC requires credentialed access and was not run. The sparse MIMIC demo and five-participant movement experiment are archived supplementary results, excluded from the revised main evidence. The manuscript contains the proofs and related-work positioning.',
    '', '## Reproduce','Run `bash run_all.sh` with `configs/execution.json` set to revision. Source dependencies and protocols are pinned. Every model selection uses validation only; no test frontier is used for tuning. See provenance.json for the exact contract.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
