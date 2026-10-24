# Methods and mathematical specification

This study asks whether consent-aware information restrictions and calibrated
review retain useful prediction while enforcing auditable flow and capacity
constraints. EHR plus system-graph analysis is primary; skeletons are secondary.
No raw video is required for these central tests.

## Endpoints and split design

The EHR endpoint is **in-hospital death in the next 48 hours**, evaluated at ICU
hours 6, 12 and 24. Features require both event and storage timestamps no later
than the landmark. Training-only imputation/scaling and patient-disjoint
training/calibration/test partitions prevent ordinary feature and patient leakage.
Metadata, full EHR, no-labs and no-vitals models use fixed C=1 logistic regression.
The public data contain no observed consent decisions; restrictions are experimental.

Random splits use 50% training, 25% calibration and 25% test patients. A separate
coarse anchor-era stress test trains/calibrates on the earlier era and tests on
later eras. This is not exact calendar chronology: patient dates are shifted.
One institution precludes an external-hospital evaluation.

Skeleton windows use 32 observed frames, stride 8, and the last observed frame's
impact label. Features exclude filename labels, subject IDs and activity codes.
Every test participant is held out with another participant reserved for
calibration; three remain for fitting. Both views/repetitions stay with the subject.
Trial 3 is a repetition-order stress subset, not a calendar holdout.

The compact ST-GCN has two spatial/temporal blocks, 32/64 channels and a 3-frame
temporal kernel, trained for 20 epochs with Adam at .001. Graph, topology-free
and degree-preserving rewired variants share training budgets and seeds. The
summary baseline uses per-joint means, variability and absolute frame differences.

## Consent and oversight

The system is a typed directed temporal graph $G_t=(V,E_t,\tau)$ with patient,
purpose, provenance and edge-time attributes.

$$A^{allowed}_{i,p}=A\odot M_{i,p}.$$

Permissions are deny-by-default. Withdrawal invalidates future use of a source
and provenance-linked stored derivatives for the revoked purpose. It does not
claim physical erasure or machine unlearning. Every persistent sink, including
an EHR write, is blocked when storage of video-derived information is denied.
Ephemeral bedside alerts can remain reachable.

$$\forall\pi\in\mathcal P(S,D),\quad\pi\cap H\ne\varnothing.$$

The exact review test deletes H and enumerates surviving simple S-to-D paths.
Any reachable path has a simple witness; thus this is exact for the finite static
graph. Time-respecting reachability is additionally checked. Endpoint-protected
minimum vertex cuts, degree and betweenness identify structural candidates.
Removal, delay and corruption tests measure availability and exposure separately.
Passing the review constraint on a disconnected system is not useful availability.

## Statistical risk

For patient i, $L_i(\lambda)$ is the fraction of positive windows not alerted,
with zero loss when that patient has no positive windows. Alert when
$1-s_{it}\le\lambda$. Then $0\le L_i\le1$, and L is non-increasing and
right-continuous in lambda. Ties enter together.

$$\widehat\lambda=\inf\{\lambda:(n\widehat R_n(\lambda)+B)/(n+1)\le\alpha\},
\qquad B=1,\quad\alpha=.05.$$

If the set is empty, use the terminal all-alert rule lambda=1. Assumptions:
frozen model/loss; exchangeable calibration and future patient loss functions;
bounded monotone right-continuous loss; terminal loss zero. The expectation is
over calibration and a new patient, not a per-patient, subgroup, or fixed-sample
bound. This unconditional loss differs from positive-patient conditional misses,
which are reported separately. No guarantee is asserted under era, hospital,
missingness or score shift, nor after queue-based suppression of alerts.

Synthetic validation uses independent patients with correlated eight-window
trajectories. Each of three seeds has 200 repetitions per calibration size
(10,25,100); each is evaluated on 100 new patients. Monte Carlo uncertainty is
not uncertainty about clinical performance.

## Capacity

$$\sum_i\mathbb E[r_i]\le K.$$

K is reviews/hour. The expected alert fraction is estimated using held-out demo
scores. Opportunity rates 5/15/30/60 per hour, service times 2/5/10 minutes and
8-hour periods are assumptions, not observed clinical operations. The FCFS queue
has one server and a 20-place waiting buffer. Backlog, waiting, overflow and
completed review are reported separately; expected feasibility is not a hard cap.

## Uncertainty, privacy and inference limits

Three fixed seeds are 11,29,47. Patient-cluster bootstrap uses 500 replicates
for prespecified seed-11 comparisons. Paired model differences resample the same
patients. Valid bootstrap counts are reported; sparse positives and five motion
participants can make intervals unstable. No test-set model selection occurs.

Prohibited storage reachability and corrupted-descendant counts measure modeled
information exposure. They are not measured re-identification or reconstruction
leakage. Raw-versus-skeleton privacy attacks are outside this execution's revised
scope. No privacy benefit is inferred from skeletons alone.

Provider→order→patient graphs represent observed ordering relationships before
prediction, not note-access logs, communication quality, or effective teamwork.
Semi-synthetic pairing attaches unrelated demo profiles to movement participants;
it does not create additional independent people or clinical associations.

All conclusions concern framework behavior on public/demo/simulated inputs.
