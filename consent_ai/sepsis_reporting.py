from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .orx_figstyle import use_style,save,TEXT,PALETTE
from .risk import ASSUMPTIONS,LOSS
from .figures import HTML

def create(out,cfg,provenance):
    out=Path(out);m=pd.read_csv(out/'metrics.csv');ci=pd.read_csv(out/'intervals.csv');cal=pd.read_csv(out/'calibration.csv');cohort=pd.read_csv(out/'cohort_summary.csv');effects=pd.read_csv(out/'paired_effects.csv')
    fd=out/'figures';fd.mkdir(exist_ok=True);use_style()
    d=effects[(effects.site=='B')&(effects.metric=='auroc')].sort_values('delta_b_minus_a')
    fig,ax=plt.subplots(figsize=(TEXT,2.8));ax.axvline(0,color='#888',linestyle='--',linewidth=.8)
    for i,r in enumerate(d.itertuples()):
        ax.plot([r.lower,r.upper],[i,i],color=PALETTE['blue'],linewidth=1.5);ax.plot(r.delta_b_minus_a,i,'o',color=PALETTE['blue'])
    ax.set_yticks(range(len(d)),[s.replace('_',' ') for s in d.model_b]);ax.set_xlabel('Hospital B AUROC difference versus full EHR');ax.set_ylabel('Feature policy');ax.set_ylim(-.6,len(d)-.4)
    fig.subplots_adjust(left=.33,right=.97,bottom=.22,top=.95);save(fig,str(fd/'hospital_b_feature_effects'),formats=('pdf','svg','png'))
    # Keep this figure scientifically distinct: risk estimand versus workload.
    q=m[(m.seed==11)&(m.model=='full_ehr')&(m.site=='A')&m.rule.isin(['crc_unconditional','crc_positive_conditional'])]
    fig,ax=plt.subplots(figsize=(TEXT,3.1))
    colors=[PALETTE['blue'],PALETTE['orange']]
    for color,r in zip(colors,q.itertuples()):
        label='Unconditional CRC' if r.rule=='crc_unconditional' else 'Positive-patient CRC'
        ax.scatter(r.review_fraction,r.positive_patient_missed_loss,color=color,s=45,label=label)
    ax.axhline(.05,color='#777',linestyle='--',label='5% conditional-loss reference')
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Fraction of hourly opportunities alerted');ax.set_ylabel('Positive-patient missed-window fraction');ax.legend(fontsize=7,loc='upper right',frameon=True)
    fig.subplots_adjust(left=.14,right=.97,bottom=.2,top=.96);save(fig,str(fd/'risk_workload_estimand'),formats=('pdf','svg','png'))
    shutil.copy('consent_ai/sepsis_reporting.py',fd/'sepsis_reporting_source.py');shutil.copy('consent_ai/orx_figstyle.py',fd/'orx_figstyle.py')
    lines=['# Open-cohort follow-up: consent, statistical risk and review demand','',
    '**REAL OPEN DATA + CONSTRUCTED POLICY. No clinical validation or patient-benefit claim.**',
    'This follow-up retains the original MIMIC-demo experiment and adds a larger approval-free cohort because that endpoint contained only two positive patients. It does not change the earlier endpoint or select a model on test results.','',
    '## Cohort and task',
    'A uniform, label-blind sample of 3,000 patients per public hospital was drawn from the complete PhysioNet/CinC 2019 listings using seed 20261023. Files are pinned by source ETags/MD5 and SHA256. The sample is a bounded evaluation, not the full 40,336-patient release.']
    for r in cohort.itertuples():lines.append(f'- Hospital {r.site}: {r.patients} patients, {r.hours} observed hourly rows, {r.positive_patients} positive patients and {r.positive_hours} positive rows.')
    lines += ['The outcome is the provided early-warning SepsisLabel at each hour. Labels were already shifted six hours early by the dataset authors; we do not shift them again. They include hours after the first positive label, so this is not an incident-only or strictly pre-onset forecasting evaluation.',
    'Use current/past rows only. Features are forward-filled values, cumulative means and observed counts. Absolute dates and laboratory storage times are unavailable; no exact calendar holdout or store-time availability guarantee is asserted. Hospital A supplies training/calibration/internal test (60/20/20%), while all sampled B patients are held out externally.','',
    '## Discrimination and probability quality',
    'Fixed C=1 logistic regressions compare metadata, full EHR and experimental feature restrictions. Brier is a proper probability score, not a standalone calibration measure; reliability-bin data are reported separately. The following are three-seed means. Seed-11 patient-cluster 95% bootstrap intervals and paired effects are linked separately.']
    a=m[m.rule=='fixed_0.5'].groupby(['site','model'])[['auroc','average_precision','brier']].mean()
    for (site,name),r in a.iterrows():lines.append(f'- Hospital {site}, {name}: AUROC {r.auroc:.4f}; average precision {r.average_precision:.4f}; Brier {r.brier:.4f}.')
    lines += ['', 'See `intervals.csv`, `paired_effects.csv`, `reliability.csv`, `subgroups.csv`, `splits.json` and `predictions.csv.gz`. Intervals use 500 whole-patient resamples, with identical resampling for paired differences. No selection or tuning used hospital B.',
    '', '## The important distinction: which risk is controlled?',
    f'**Base patient loss:** {LOSS}. B=1, alpha=.05. **Unconditional CRC** calibrates on all calibration patients. **Positive-patient CRC** calibrates on the subset with any positive window; its guarantee additionally requires exchangeability conditional on positive-patient status. Both frozen rules apply to every test patient.',
    f'**Assumptions for every CRC number below:** {ASSUMPTIONS} The positive-conditioned rule uses conditional patient exchangeability. No guarantee is asserted for hospital B or missingness shift.',
    'The finite-sample rule is lambda_hat = inf{lambda: (n R_hat(lambda)+1)/(n+1) ≤ .05}, with an all-alert terminal fallback. Counts and thresholds for every fit are in `calibration.csv`. Expected risk is not a guarantee for an individual or fixed realized calibration sample.']
    for r in m[(m.seed==11)&(m.model=='full_ehr')&m.rule.isin(['crc_unconditional','crc_positive_conditional'])].itertuples():
        lines.append(f'- Seed 11, hospital {r.site}, {r.rule}, n={int(r.n)}, B=1, alpha=.05: patient-average missed loss {r.patient_missed_loss:.4f}; positive-patient conditional missed loss {r.positive_patient_missed_loss:.4f}; hourly sensitivity {r.sensitivity:.4f}; review fraction {r.review_fraction:.4f}; false alerts per 24 observed patient-hours {r.false_alerts_per_24_observed_hours:.2f}. Assumptions and the distinct estimands are as defined immediately above; hospital B is a shift test.')
    lines += ['If non-positive patients have zero loss, unconditional risk equals positive-patient prevalence multiplied by conditional missed loss. A small unconditional risk can therefore coexist with high conditional misses. This is a mathematical limitation of the chosen estimand, not a failure of the CRC theorem.',
    '', '## Capacity and the research hypotheses',
    '`capacity.csv` evaluates the estimated hourly alert fraction against K=2/5/10/20/40/60 and simulated opportunity rates 5/15/30/60 per hour. These are scenario parameters, not observed nursing workloads. The earlier `../queue_sweep.csv` additionally measures burstiness, waiting, overflow and completed reviews. No post-calibration alert suppression is permitted while retaining the original guarantee.',
    '**H1:** supported for the constructed graph: exact checks expose review bypasses, unsafe storage paths and cached-derivative reuse. Multi-source provenance now includes both EHR and video parents; revoking either blocks merged derivatives. This does not validate an actual hospital network.',
    '**H2:** partially testable here: compare predictive quality of permitted feature sets and the risk/workload consequence of explicit loss definitions. Structural privacy constraints can preserve ephemeral alerts, but useful clinical performance and feasible human review are not established by ROC or CRC alone.',
    '**H3:** the system graph supplies auditable reachability/cut information absent from predictive accuracy. The separate five-participant skeleton ablation remains too small to establish general predictive graph superiority. No provider-graph survival effect is claimed.',
    '', '## Limits',
    'This is a sample of a retrospective benchmark, with simulated consent and capacity. It does not establish clinical effectiveness, voluntary consent, reviewer competence, reconstructed-data privacy, model unlearning, or prospective benefit. The benchmark labels can reflect existing sepsis; sparse hourly measurements and unavailable acquisition/store timing constrain causal availability claims. Subgroup analyses are descriptive and not multiplicity-adjusted. Confidence intervals quantify patient resampling within a sample, not all data/label/model uncertainty.',
    '', '## Reproduce',
    'The single `run_all.sh` entry point uses the pinned protocol and sample manifest. `python -m consent_ai.run --stage all` regenerates both the initial public/demo evaluation and this follow-up; `--stage sepsis` regenerates this stage. All figures are PDF/SVG/PNG and every analytical table is CSV.',
    'Primary source: [PhysioNet/CinC 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/), Reyna et al., DOI 10.13026/v64v-d857, CC BY 4.0. Statistical source: [Conformal Risk Control, Eq. 4 and Theorem 1](https://www.alphaxiv.org/pdf/2208.02814?page=2).',
    '', '## Provenance','```json',json.dumps(provenance,indent=2),'```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    views=[]
    for site in ['A','B']:
        for metric,unit in [('auroc','AUROC'),('brier','Brier score (lower is better)')]:
            points=[]
            for r in m[(m.seed==11)&(m.site==site)&(m.rule=='fixed_0.5')].itertuples():
                c=ci[(ci.site==site)&(ci.model==r.model)&(ci.metric==metric)].iloc[0]
                points.append(dict(label=r.model.replace('_',' '),value=float(getattr(r,metric)),low=float(c.lower),high=float(c.upper)))
            views.append(dict(label=f'Hospital {site} · {unit}',unit=unit,points=points,note='REAL OPEN DATA. Prespecified seed 11; whiskers are 95% percentile intervals from 500 patient-cluster bootstrap replicates. A is internal patient holdout; B is an external-site shift test. See report for all three seeds.'))
    for metric,unit in [('positive_patient_missed_loss','Positive-patient missed-window fraction'),('review_fraction','Fraction of hourly opportunities alerted')]:
        pts=[]
        for r in m[(m.seed==11)&(m.model=='full_ehr')&m.rule.isin(['crc_unconditional','crc_positive_conditional'])].itertuples():
            value=float(getattr(r,metric));pts.append(dict(label=f'{r.site} '+('all-patient CRC' if r.rule=='crc_unconditional' else 'positive CRC'),value=value,low=value,high=value))
        views.append(dict(label=unit,unit=unit,points=pts,note='Seed-11 frozen full-EHR policies; point estimates only. Alpha=.05, B=1. Unconditional loss and positive-patient conditional loss are different estimands. See calibration.csv for n and threshold. Hospital B has no asserted exchangeable CRC guarantee.'))
    html=HTML.replace('__DATA__',json.dumps(views,allow_nan=False))
    html=html.replace('EHR results use the 100-patient MIMIC-IV demo.','Primary EHR results use 6,000 public sepsis-benchmark patients. The earlier 100-patient MIMIC demo is retained as a separate limited check.')
    html=html.replace('EHR outcome: in-hospital death within 48 hours at ICU hours 6, 12 and 24. Both observation and storage timestamps precede prediction.','Sepsis endpoint: the provided hourly early-warning target, already shifted six hours early; it also includes later positive hours. Features use current/past rows only. Store timestamps are unavailable.')
    html=html.replace('Random patient holdout and a separate coarse anchor-era shift test. Date de-identification prevents a strict calendar-time claim. No external hospital exists in this dataset.','Hospital A is split 60/20/20% by patient; all sampled hospital B patients form an external-site shift test. Absolute dates are unavailable, so no calendar holdout is claimed.')
    html=html.replace('Seed ranges in this chart are not confidence intervals.','Whiskers in discrimination views are patient-bootstrap confidence intervals; risk/workload views show point estimates only.')
    html=html.replace('ehr_metrics.csv','metrics.csv').replace('ehr_paired_effects.csv','paired_effects.csv').replace('capacity_sweep.csv','capacity.csv').replace('queue_sweep.csv','../queue_sweep.csv').replace('movement_metrics.csv','../movement_metrics.csv')
    html=html.replace('<a href="https://physionet.org/content/mimic-iv-demo/2.2/">MIMIC-IV demo</a>','<a href="https://physionet.org/content/challenge-2019/1.0.0/">PhysioNet 2019 sepsis</a> · <a href="../REPORT.md">Initial demo/skeleton report</a>')
    html=html.replace('select the smallest lambda satisfying','for the selected patient population (all patients or positive patients), select the smallest lambda satisfying')
    (out/'index.html').write_text(html)
    print('SEPSIS_REPORT_READY',str(out/'REPORT.md'),flush=True)
