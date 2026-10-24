# Consent-constrained clinical AI

A reproducible research prototype investigating whether patient/purpose consent,
information-flow constraints and risk-calibrated human review can coexist with
useful predictions and finite review capacity.

**Research only. No clinical validation or patient-benefit claim.** The primary
empirical evaluation uses **MIMIC-IV demo data**. Review queues and system graphs
are simulated. Public UP-Fall skeletons provide a secondary modality check.

## Results

- [Interactive results and discussion](results/index.html) — self-contained HTML;
  download and open locally, or serve this checkout with `python3 -m http.server`.
- [Research report](results/REPORT.md)
- [Methods, math and limitations](METHODS.md)
- [Data sources and access](DATA_CARD.md)

The results directory contains CSV tables, patient splits, confidence intervals,
paired effects, PDF/SVG/PNG figures and run provenance. Raw inputs and derived
research data are kept distinct from model results. GitHub does not execute HTML
in its file viewer; use the local server or deploy the static results directory.

## Reproduce

Install Python 3.13 and [uv](https://docs.astral.sh/uv/), then:

```bash
bash run_all.sh
```

This installs the locked dependencies, runs the mathematical and leakage tests,
downloads/checks public inputs, evaluates all prespecified models and regenerates
the report, every CSV and every figure. No GPU, API key, or paid service is needed.
The three-seed ST-GCN ablations are the longest step. Inputs are cached after
their first download. Set `CONSENT_DATA_ROOT` or `CONSENT_RESULTS_DIR` to relocate
data or outputs.

On macOS, if a Desktop-local virtual environment hangs, create one outside
Desktop and symlink it into this checkout before running:

```bash
uv venv ~/.venvs/consent_clinical_ai --python 3.13
ln -s ~/.venvs/consent_clinical_ai .venv
uv sync --locked
```

Do not reuse an environment while another checkout is modifying its dependencies.

## Study components

| Component | Evidence | Boundary |
|---|---|---|
| EHR risk and consent variants | Patient-disjoint demo-data predictions | Not clinical efficacy |
| Typed temporal graph | Exact path/cut and withdrawal checks | Constructed graph, not a deployed-system audit |
| Conformal risk control | Patient-level finite-sample calibration and synthetic validation | Expectation under assumptions, not individual safety |
| Review capacity | Expected-load sweep plus FCFS queues | Simulated capacity/service times |
| Skeleton ST-GCN | Summary, no-topology and rewired ablations | Five public participants |
| Provider/order graph | Pre-cutoff ordering relationships | Not note reading or proven teamwork |

## Code map

- `consent_ai/graph.py`: consent mask, provenance withdrawal, exact static and temporal paths.
- `consent_ai/risk.py`: patient loss, CRC, capacity and queue.
- `consent_ai/ehr.py`: causal landmark prediction and provider/order graph.
- `consent_ai/movement.py`: compact ST-GCN and matched ablations.
- `consent_ai/simulation.py`: interventions, capacity sweep, synthetic checks and pairing.
- `consent_ai/evaluation.py`: metrics, cluster bootstrap and paired comparisons.
- `consent_ai/figures.py`: publication figures and interactive HTML.
- `configs/protocol.json`: frozen endpoints, seeds and hyperparameters.
- `tests/`: mathematical identities, counterexamples and temporal leakage checks.

## Data and licensing

UP-Fall skeleton derivative: Koffi et al., [Zenodo 12773013](https://zenodo.org/records/12773013),
CC BY 4.0. MIMIC-IV demo: Johnson et al., [v2.2](https://physionet.org/content/mimic-iv-demo/2.2/),
PhysioNet Open Access license supplied with the data. Dataset licenses remain
applicable to redistributed data and derived outputs. See `DATA_CARD.md`.

SAFER is skipped pending access approval. Full MIMIC-IV and NTU require the user
to obtain access; the code never accepts agreements or bypasses access controls.

## Scientific references

- Angelopoulos et al., [Conformal Risk Control](https://www.alphaxiv.org/pdf/2208.02814), Eq. 4 and Theorem 1.
- Yan et al., [Spatial Temporal Graph Convolutional Networks](https://www.alphaxiv.org/pdf/1801.07455). This project uses a compact adaptation.
- Koffi et al., [Improved 3D Skeletons UP-Fall Dataset](https://www.alphaxiv.org/pdf/2502.19048).
- Johnson et al., [MIMIC-IV demo v2.2](https://doi.org/10.13026/dp1f-ex47).
