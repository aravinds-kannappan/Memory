from pathlib import Path
import json,shutil
import pandas as pd
from .risk import ASSUMPTIONS,LOSS

def write_report(out,cfg,provenance):
    out=Path(out);ehr=pd.read_csv(out/'ehr_metrics.csv');motion=pd.read_csv(out/'movement_metrics.csv');graph=pd.read_csv(out/'graph_scenarios.csv');mc=pd.read_csv(out/'crc_monte_carlo_summary.csv');audit=json.loads((out/'ehr_audit.json').read_text());ca=pd.read_csv(out/'ehr_calibration.csv')
    e=ehr[(ehr.regime=='random')&(ehr.policy=='fixed_0.5')].groupby('model')[['auroc','average_precision','brier']].mean()
    m=motion[motion.subset=='all_heldout'].groupby('model')[['auroc','brier']].mean()
    lines=['# Consent-constrained clinical AI: executed research report','',
    '**Scope:** EHR plus mathematical consent/oversight evaluation is primary; public skeletons are secondary, following the user’s revised scope. No clinical validation or patient benefit is claimed.','',
    '## What ran',f'- **DEMO DATA:** {audit["raw_patients"]} source MIMIC-IV demo patients, {audit["eligible_patients"]} eligible patients and {audit["landmarks"]} landmark decisions; {audit["positive_landmarks"]} positive outcome windows. See `ehr_audit.json`.',
    '- **OPEN DERIVED SKELETONS:** all five archives in Zenodo 12773013; compact ST-GCN, summary, topology-free and shuffled-edge comparators; participant-held-out evaluation.',
    '- **SIMULATION:** exact graph policies, cached-derivative withdrawal, structural stress tests, expected-load and FCFS queue sweeps.',
    '- **SYNTHETIC MATHEMATICAL CHECK:** CRC Monte Carlo with independent patient trajectories and correlated within-patient windows, plus deliberate score shift.',
    '- **SEMI-SYNTHETIC:** observed skeleton windows paired with unrelated demo EHR profiles; no real patient linkage or causal association is inferred.',
    '- **NOT RUN:** full credentialed MIMIC-IV, NTU, SAFER, raw-video pose extraction/identity attacks, clinical bed-exit/delirium endpoints, external-hospital or prospective clinical validation.','',
    '## H1: structure reveals risk',
    'The implemented exact path check identifies unreviewed action paths in the permissive graph. Removing bypass edges and requiring physician review eliminates them. Consent removes persistent-storage paths while retaining bedside alert and ephemeral action reachability. This verifies the finite graph, not the completeness of a clinical system representation.',
    'See `graph_scenarios.csv`, `bypass_witnesses.json`, `graph_centrality.csv`, `graph_stress.csv` and `withdrawal.csv`. Direct source-edge masking leaves cached derivative reuse possible; provenance-aware masking blocks it. Corruption exposure counts descendants, not demonstrated changes in clinical predictions.',
    '', '## H2: constraints preserve utility',
    'Three-seed mean discrimination/calibration on random held-out demo patients (the seed range is not an independent-patient confidence interval):']
    for name,r in e.iterrows():lines.append(f'- {name}: AUROC {r.auroc:.4f}; average precision {r.average_precision:.4f}; Brier {r.brier:.4f}.')
    lines += ['', 'Consent variants are separately fitted using only permitted features. They isolate an information-restriction tradeoff; the public dataset contains no observed patient consent decisions. Patient-level intervals and paired effects are in `ehr_intervals.csv` and `ehr_paired_effects.csv`.',
    'Review workload and privacy exposure remain separate from model discrimination. Queue overload or an all-alert fallback prevents interpreting structural compliance as useful operational oversight. Useful clinical performance is not established by this demo.',
    '', '## H3: graph structure must earn its place',
    'Secondary skeleton results, averaged across three training seeds after participant-held-out predictions are pooled:']
    for name,r in m.iterrows():lines.append(f'- {name}: AUROC {r.auroc:.4f}; Brier {r.brier:.4f}.')
    lines += ['Use `movement_paired_effects.csv` for paired differences and `movement_intervals.csv` for participant-bootstrap intervals. Five participants are insufficient to establish general graph superiority; overlapping windows do not increase the independent sample size. Models have the same graph training budget and fixed splits. No test-set winner was selected.',
    '', '## Mathematical definitions and statistical interpretation',
    '1. **Typed graph:** node/edge types, provenance, patient IDs, event timestamps and latency are explicit. Time-respecting reachability is separate from static paths.',
    '2. **Consent:** A_allowed(i,p) = A ⊙ M(i,p). Tests check matrix equality, purpose and patient isolation, withdrawal and derivative reuse.',
    '3. **Review:** enumerate all simple sensitive-source→action paths after deleting mandatory review nodes. Test equivalence exhaustively on all 64 forward DAGs over four vertices. Paths with cycles have simple witnesses, so a simple-path check is sufficient for static reachability. Availability is separately measured.',
    '4. **Capacity:** sum_i E[r_i] ≤ K per hour. The empirical held-out score distribution estimates alert probabilities; arrival rates and service times are simulated. An expectation constraint is not a queue-length guarantee.',
    '5. **CRC:** lambda_hat = inf{lambda: (n R_hat(lambda)+B)/(n+1) ≤ alpha}; alpha=0.05, B=1. Use exact score knots, inclusive ties, and lambda=1 if the feasible set is empty.',
    f'**Loss:** {LOSS}. This is an unconditional patient-average loss, not conditional false-negative rate. Report positive-patient missed fraction and sensitivity separately.',
    f'**Assumptions adjacent to CRC results:** {ASSUMPTIONS}',
    f'Calibration sizes and fallback status are recorded for every model in `ehr_calibration.csv`. Observed n values: {sorted(ca.n.unique().tolist())}. The secondary movement calibration uses n=1, so the finite-sample correction is 1/2 and forces the terminal all-alert fallback at alpha=.05.',
    '', '### Synthetic validation of the statistical procedure',
    'The following checks use synthetic data only. Each n/shift setting has 600 independent calibration repetitions across three seeds, evaluated on 100 fresh synthetic patients per repetition; 95% intervals are Monte Carlo uncertainty, not a clinical risk bound.']
    for r in mc.itertuples():lines.append(f'- n={r.n}, shifted={r.shifted}, B=1, alpha=.05: mean patient missed loss {r.mean_risk:.4f}, Monte Carlo interval [{r.mc_lower:.4f}, {r.mc_upper:.4f}]. Loss and assumptions as defined immediately above; shifted results have no asserted CRC guarantee.')
    lines += ['', '## Causality, holdouts, subgroups and uncertainty',
    '- EHR event and store timestamps must precede the 6/12/24-hour landmark. Unit tests perturb later and late-recorded events and verify unchanged past features. Outcomes are kept out of the feature matrix.',
    '- Train/calibration/test splits are patient-disjoint. Imputation/scaling fit training patients only. Three fixed seeds are reported; seed 11 has 500 patient-bootstrap replicates and paired comparisons. Bootstrap intervals can be unreliable with sparse positive patients; valid replicate counts are supplied.',
    '- Coarse anchor-era holdout and missingness shift are explicit stress tests. Anchor-year groups are not exact chronology; a strict calendar holdout cannot be claimed from these de-identified dates. A second hospital is unavailable.',
    '- Gender and race subgroups include support counts; low-support cells are labeled. Group comparisons are exploratory, without multiplicity-adjusted inference. Motion participant/trial subgroups are in fold metrics; demographic attributes were not inferred.',
    '- No event-rate-per-day estimate is computed from sparse EHR landmarks or skeleton frames with unverified frame timing. False-positive fraction and false-alert counts use explicitly defined opportunities.',
    '', '## Privacy and the provider graph',
    'Privacy outputs count prohibited storage reachability and cached-derivative reuse only. These do not measure membership inference, re-identification or reconstruction. No raw-versus-skeleton privacy benefit has been measured.',
    'Provider→order→patient edges use pre-cutoff provider identifiers. The projected graph is descriptive and is not an access-log reproduction of Lu and Ma, a communication graph, or an outcome association. No predictive provider-graph model was fitted, so its structural centralities are not presented as predictive ablations.',
    '', '## Limitations and evidence against overclaiming',
    'The demo is small and selected; actual review capacity, reviewer competence, voluntary consent and clinical usefulness are unmeasured. The system graph is constructed, not audited from a deployed system. Synthetic queue arrivals and semi-synthetic pairing cannot identify clinical causal effects. Withdrawal invalidates future use in this implementation; it is not physical erasure or model unlearning. Real-time alerts may require ephemeral clinically meaningful information even when storage is denied. EHR writes are treated as storage and therefore blocked for video-derived denied storage purposes.',
    'Small calibration sets can produce either trivial all-alert fallback or a permissive rule when the unconditional loss sees few positive patients. Neither proves a low conditional false-negative rate. Post-calibration dropping, review delays and distribution shift invalidate an unqualified risk-control interpretation.',
    '', '## Reproduction and outputs',
    'Run `bash run_all.sh`. Dependencies are locked in `uv.lock`; configuration is in `configs/protocol.json`. Inputs, publisher checksums and URLs are in `data_manifest.json`. The local experiment source snapshot and run identifier are recorded in `provenance.json`. Source includes unit tests, not just assertions in evaluation code.',
    'Every analytical table is a CSV in this folder. Figure 1 is an aggregation of implemented allowed graph nodes; Figure 2 shows the actual consent graph (dashed grey edges are denied storage paths). Figures are exported as PDF, SVG and PNG. `index.html` is a self-contained interactive results page.',
    '', '## Sources',
    '- Johnson et al. MIMIC-IV demo v2.2: https://physionet.org/content/mimic-iv-demo/2.2/ (DEMO DATA).',
    '- Koffi et al. Derived 3D skeleton UP-Fall data: https://zenodo.org/records/12773013 (CC BY 4.0); methods: https://www.alphaxiv.org/pdf/2502.19048.',
    '- Angelopoulos et al. Conformal Risk Control, Eq. 4 and Theorem 1: https://www.alphaxiv.org/pdf/2208.02814?page=2.',
    '- Yan et al. ST-GCN: https://www.alphaxiv.org/pdf/1801.07455. Compact adapted architecture, not benchmark reproduction.',
    '', '## Run provenance', '```json',json.dumps(provenance,indent=2),'```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

