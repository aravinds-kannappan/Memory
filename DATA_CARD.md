# Data access, provenance and limitations

## Inputs used

**MIMIC-IV Clinical Database Demo v2.2 — DEMO DATA.**
Johnson et al., DOI [10.13026/dp1f-ex47](https://doi.org/10.13026/dp1f-ex47).
The public subset contains 100 de-identified patients. Selected tables are
patients, admissions, labs, prescriptions, provider orders, ICU stays, charted
events, input events and item dictionary. Public license text and SHA256SUMS
are retained. No free-text notes are available; no note-access events are invented.
The demo is not representative of the full database or another hospital.

**UP-Fall 3D skeleton derivative — OPEN DATA.**
Koffi et al., [Zenodo 12773013](https://zenodo.org/records/12773013), CC BY 4.0.
All five SUBJECT archives are included in acquisition; publisher MD5s and local
SHA256s are verified. Coordinates were extracted by the dataset authors using
MediaPipe; no new pose extraction is claimed. Original attribution:
Martínez-Villaseñor et al., UP-Fall Detection Dataset, Sensors 19(9), 1988 (2019),
DOI 10.3390/s19091988. Changes here: header normalization, root centering,
observed-prefix windowing, participant splitting, and derived predictions.
The final CSV label is checked as binary even for source headers LLABEL and 0.
A malformed camera prefix is not silently converted into a known camera ID.

This derivative is small, selected and dominated by staged falls/impact frames;
it is not the full original UP-Fall distribution. Clinical bed exit, loss of
balance and future-fall benefit cannot be inferred from these impact labels.
Original RGB camera downloads were quota-limited when checked. A raw-video
privacy comparison was deferred under the user's revised research scope.

`results/data_manifest.json` records exact files, URLs, bytes and checksums.
`results/movement_file_audit.csv` records every loaded movement file.
`results/ehr_splits.json` and `results/movement_splits.json` record split membership.
Downloaded data and derived data retain the source licenses.

## Not available / not used

- **SAFER-Activities:** user requested skipping while approval is pending. The
  current Hugging Face repository is gated (403 without approval), despite its
  research-use CC BY-NC-SA 4.0 license. No agreement was accepted by this pipeline.
- **NTU RGB+D:** TODO(user): register at https://rose1.ntu.edu.sg/dataset/actionRecognition/,
  accept the release agreement yourself, then place official `.skeleton` files
  under `data/raw/ntu/`. `consent_ai.data.ntu_skeleton` parses the official format;
  the present reported experiment does not claim NTU training or evaluation.
- **Full MIMIC-IV v3.1:** TODO(user): complete PhysioNet credentialing/CITI training,
  sign the data use agreement at https://physionet.org/content/mimiciv/3.1/, and
  place licensed `hosp/` and `icu/` directories under `data/raw/mimiciv/`.
  The shared-schema EHR loader selects that directory when present; existing
  public-demo results must be replaced and relabeled after any full-data run.
- No linked patient video/EHR, external hospital, or prospective clinical cohort.

## Redistribution

The GitHub deliverable includes public inputs only, their license/provenance,
and clearly labeled derived research results. Credentialed data must remain
local and must never be committed. The default ignore rules exclude all data;
only this reviewed public subset is explicitly added for this release.
