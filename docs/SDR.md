# Design Solution Definition: EMG Decoding Pipeline

> **Project:** EMG Decoder V2
> **Document ID:** `DSD-001-EMG-DECODING`
> **Version:** `0.1.0` · **Status:** `Draft`
> **Project phase:** `Proof of Concept`
> **Last modified:** `2026-10-18`
> **Repository path:** `docs/SDR.md`

---

## Document control

### Roles and sign-off

| Role | Name | Affiliation / Handle | Date | Signature / Review link |
|---|---|---|---|---|
| Author(s) | Guilherme Costa-Ferreira| TL / `GuilhermeCF`| | |
| Author(s) | David Gorgiev| EMG Member / `dgorgiev`| | |
| Author(s) | Luca Tellegen| EMG Member / `lucawltepfl`| | |
| Reviewer (technical) | `<TBD>` | | | |
| Concurrence (SE Reviewer) | `<TBD>` | | | |
| Concurrence (Quality management reviewer) | `<TBD>` | | | |
| Approval (Project Manager) | `<TBD>` | | | |

> Note : Link approval to any relevant PR (Pull Request)
> Affiliation/Handle : Role/ Username on Github

### Modification log

| Version | Date | Author(s) | Main changes | PR / Commit |
|---|---|---|---|---|
| 0.1.0 | 2026-10-18 | Guilherme Costa-Ferreira | First draft filled from repository state (`docs/sdr` branch) | `cd7b1a3` |

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Requirements and traceability](#2-requirements-and-traceability)
3. [Concept justification](#3-concept-justification)
4. [Traded-off concepts](#4-traded-off-concepts)
5. [Quality management and compliance](#5-quality-management-and-compliance) /!\ THIS SECTION IS NOT FINALISED /!\
6. [Licensing](#6-licensing) /!\ THIS SECTION IS NOT FINALISED /!\
7. [Open issues and future work](#7-open-issues-and-future-work)
8. [References](#8-references)
9. [Appendices](#9-appendices)

---

## 1. Introduction

### 1.1 Purpose and scope

This Design Solution Definition (DSD) is at a `Proof of concept` stage
validated on `<date>`. It justifies the concept selected for the `EMG decoding pipeline V2`.

It covers:

- the problem the subsystem must solve
- the retained concept: architecture, components, and design choices
- the concepts that were considered and not retained
- risks, quality, and compliance implications.

**Out of scope:**
- EMG acquisition hardware (analog front-end, electrodes, ADC board) - covered by the BMI hardware team.
- Embedded firmware and on-MCU deployment - assessed separately in [`docs/adapt_to_board.md`](adapt_to_board.md) and on the [Firmware Repository](https://github.com/N-Pulse/BMI-EMG-Firmware/tree/i2c_adc)
- EEG decoding and prosthesis actuation/control - also on their own [EEG Repository](https://github.com/N-Pulse/BMI-SOFT-EEG-Decoding)
- Previous decoding models and research

### 1.2 Responsibility and change authority

- **Document owner:** Guilherme Costa-Ferreira, Decoding Team
- **Change authority:** Decoding TL (repository maintainer). Changes are proposed through meetings and then merged through a reviewed pull request into `main`, following `.github/pull_request_template.md`.

### 1.3 Abbreviated terms and definitions

| Term | Definition |
|---|---|
| DSD | Design Solution Definition |
| EMG / sEMG | (Surface) electromyography |
| EPN612 | EMG-EPN-612 public dataset (Myo armband, 8 ch, 200 Hz, 612 users, 6 gestures) |
| BIDS | Brain Imaging Data Structure, used for the in-house archive |
| XDF | Extensible Data Format (LSL recordings from the in-house acquisition) |
| DoF | Degree of freedom of the hand/wrist |
| MAV, RMS, WL, ZC, SSC… | Time-domain EMG features (mean absolute value, root mean square, waveform length, zero crossings, slope sign changes) |
| SVM | Support Vector Machine (linear, `LinearSVC`) |
| DT / RF | Decision tree / Random forest |
| CNN | Convolutional neural network |
| TCNN | Temporal CNN, the V2 model based on Atzori et al. (2016) |
| TTV | Train / Test / Validation split |
| ICA | Independent Component Analysis |
| κ (kappa) | Cohen's kappa score, agreement corrected for chance |
| MAKE Archive | In-house storage for recordings and trained models |
| OvA | One-versus-all classification |
| MCU | Microcontroller unit (target: STM32WBA65RI) |

## 2. Requirements and traceability

| Requirement ID | Description (derived) | Verification method | Status |
|---|---|---|---|
| REQ-SW-001 | The pipeline shall ingest in-house recordings (XDF → BIDS) and the EPN612 public dataset through a single download/load interface. | D | met (downloader subpackage) |
| REQ-SW-002 | The decoder shall classify the gesture set {noGesture, fist, rotate wrist right, rotate wrist left, open, pinch}. | T | not met |
| REQ-SW-003 | The decoder should be tested on windows of 25-200 ms | T | not met |
| REQ-SW-004 | The model + preprocessing shall fit the STM32WBA65RI (2 MB Flash, 512 KB SRAM, 100 MHz Cortex-M33). | A | not met |
| REQ-SW-005 | The integrated model should infere in less than 200 ms | T | not met |
| REQ-SW-006 | Evaluation shall be subject-disjoint (train/val/test split by participant). | I, T | met at prune subpackage |
| REQ-SW-007 | Training data shall be adapted to the deployment acquisition domain (16-bit ADC, 2000 Hz, 20–500 Hz band). | T | partial at prune subpackage|
| REQ-SW-008 | Online recognition shall report classification metrics, robustness metrics and time metrics. | T | partial met offline (`scripts/evaluation/`) |
| REQ-SW-009 | Experiments shall be reproducible from config files with fixed seeds. | I | partial |
| REQ-SW-010 | Only reviewed and refactored code should live in the subpackages, the rest shall live at scripts | D | partial |
| REQ-SW-011 | The model should outperform in classification and robustness metrics the V1 model | D | not met |

> **Verification methods:** (D)emonstration, (T)est, (M)easurement, (I)nspection, (A)nalysis

---


## 3. Concept justification

The design aims at a decoder that is (1) accurate enough on a small gesture set, (2) cheap enough to run on a low-compute chip, and (3) trainable today on public data while the in-house dataset grows. Every element below follows from those three constraints.

### 3.1 Solution overview

The main concept for V2 is to replace the V1 hand-crafted feature + classical classifier pipeline with an **end-to-end Temporal CNN (TCNN)** based on Atzori et al. (2016), trained on the **EMG-EPN-612 open-source dataset adapted to the deployment board's acquisition domain**. EPN612 is a stepping stone: it lets us build, train and understand a first implementation of the model now, so that the pipeline is ready once the Data Acquisition team has collected enough in-house recordings with our own gesture set. The EPN-trained model is then meant to be fine-tuned on in-house data rather than retrained from scratch. The model is first evaluated as a raw baseline, then improved through a controlled preprocessing study, and finally **reduced for the embedded target** following the Wang et al. (2020) pipeline (sampling rate, window size, channel silencing, quantisation). In parallel, a **science track** studies the learned CNN filters to understand and later improve the architecture.

 ![figure](v2_diagram.png)

*Figure 3.1: High-level architecture of the EMG decoding pipeline V2.*

### 3.2 Main design choices

#### 3.2.1 Temporal CNN (Atzori et al. 2016) as the V2 model

- **Decision:** Implement in PyTorch the five-block CNN of Atzori et al. (2016) (`scripts/cnn/cnn.py`): a first convolution spanning all electrodes, 3×3 and 5×5 convolutions with average pooling, a temporal convolution (5×1 or 9×1) and a 1×1 classification convolution averaged over time.
- **Rationale:** It is a simple, well-documented EMG architecture, a good starting point to compare against V1, and its small size makes later embedded reduction realistic. Learning features directly from the signal removes the hand-crafted feature step that dominated V1 compute.
- **Requirements addressed:** REQ-SW-002, REQ-SW-011
- **Consequences / limitations:** Adds PyTorch to the environment (version constraints to be reported, see OI-02). Needs more data than V1 and is less interpretable, which motivates the science track (3.2.6). Success criterion for this step is code-level: it runs without crashing on every team machine.

#### 3.2.2 Dataset adaptation to the deployment domain

- **Decision:** For V2 only the open-source EPN612 dataset is used for training. Its signals are adapted to look like what the model will see once deployed: amplitude scaling and signal resolution (8-bit Myo → 16-bit board ADC range), sampling frequency (200 Hz → board rate by resampling) and the analogue front-end (simulated first-order 20–500 Hz filter). Implemented in `src/bmiemg/data/epn612/adaptation.py`.
- **Rationale:** In-house data is still scarce and lacks the correct number of channels; training on adapted public data lets the model learn on 612 users while staying in the same numeric domain as the board.
- **Requirements addressed:** REQ-SW-007
- **Consequences / limitations:** Information above the 100 Hz source Nyquist cannot be recovered by upsampling, so the adapted data is only an approximation of board data. The EPN gesture set also differs slightly from ours, so the output layer will have to be adapted when moving to in-house data. Its similarity to local recordings must be quantified (signal similarity per task, electrode maps). We can use the incomplete but supersampled in-house data as a way to study this consequence.

#### 3.2.3 Train/Test/Val split 70/15/15 balanced by user and movement

- **Decision:** One TTV split of 70 % / 15 % / 15 %, disjoint by participant and balanced per movement (at `bmiemg/prune`). The deliverable is the split file itself.
- **Rationale:** A prosthesis must generalise to unseen users; within-subject splits overestimate accuracy. A fixed, versioned split makes all later steps comparable.
- **Requirements addressed:** REQ-SW-006, REQ-SW-009
- **Consequences / limitations:** With smaller subsets we might end up having incompete splits or false balanced situations.

#### 3.2.4 Raw baseline at a locked 150 ms window

- **Decision:** Before any preprocessing or tuning, train and evaluate the TCNN on overlapping 150 ms windows, reporting accuracy, Cohen's kappa and F1 score. Trained models are saved to the MAKE Archive and their paths recorded; training curves and model size are reported alongside.
- **Rationale:** Establishes raw model capacity. 150 ms is not yet a realistic deployment window, but fixing window and step isolates the effect of the model from the effect of segmentation. We lock the step size to a minimum of 10ms (as the firmware team has constrained that way), but final step size will be adapted to final model inference time. So for early trainning we will lock it at 25 ms step which also augments the training set.
- **Requirements addressed:** REQ-SW-003, REQ-SW-008, REQ-SW-011
- **Consequences / limitations:** Performance is expected to be modest at this stage; it is the reference for every later step.

#### 3.2.5 Preprocessing study (filtering)

- **Decision:** Quantify the effect of software preprocessing on the locked 150 ms movement-prediction task. A 50 Hz notch is tested first and kept only if it improves accuracy (EMG also has energy in that band). Other filtering parameters are then explored by grid search.
- **Rationale:** Each preprocessing step costs compute on the MCU, so it must earn its place through a measured accuracy gain and reduced inference time.
- **Requirements addressed:** REQ-SW-004, REQ-SW-008
- **Consequences / limitations:** Grid search multiplies training runs; results are tied to the adapted dataset and must be re-checked on board data. 

#### 3.2.6 Model reduction for the embedded target (Wang et al. 2020)

- **Decision:** Port the model to the board this semester by applying the Wang et al. (2020) reduction pipeline step by step, measuring the impact of each step on performance:
  1. reduce the sampling rate where possible,
  2. reduce the window size: grid search over 50, 75, 100, 150, 175 and 200 ms, crossed with a step-size grid adaptation since the step sets the decision rate and the compute per second on the MCU,
  3. silence the least important channels,
  4. quantise the weights.
  A robustness score summarising performance across realistic window and step sizes is defined for this evaluation.
- **Rationale:** Larger windows are expected to perform better, but real-time prosthesis control needs short windows; the trade-off must be measured. The pipeline is proven for EEGNet on low-power MCUs (more demanding task).
- **Requirements addressed:** REQ-SW-003, REQ-SW-004, REQ-SW-005
- **Consequences / limitations:** Each reduction may lose accuracy; deployment itself is owned by the firmware repository.

#### 3.2.7 Science track: CNN interpretability

- **Decision:** In parallel with integration, analyse the learned CNN weights, starting from the first and last layers (expected to be the most interpretable, by analogy with vision CNNs), form hypotheses on their role and verify them with ablation studies. Architecture changes inspired by work such as Liao et al. (2025) are also explored.
- **Rationale:** Understanding what the model learned guides better architectures and makes the decoder easier to trust and transfer between terms.
- **Requirements addressed:** REQ-SW-011
- **Consequences / limitations:** Experimental, with no guaranteed outcome. Requires a structured documentation base and a literature database to carry the knowledge across team turnovers. Close to the end of the semester, we might fail to complete it.

#### 3.2.8 Workflow stages

| Mode / Stage | Purpose | Entry condition | Exit condition | Documentation deliverables |
|---|---|---|---|---|
| 2.1 TCNN implementation | Reproduce Atzori et al. (2016) in PyTorch | PyTorch added to the environment | Runs without crashing on dummy data, on all team computers | Python and package version constraints |
| 2.2 Dataset adaptation | Bring EPN612 into the board domain and build the TTV split | EPN612 downloaded (`scripts/download/download_epn612.py`) | Adapted data and 70/15/15 split file exist | Dataset composition per partition and overall size; signal similarity open-source vs local per task; electrode maps |
| 2.3 Training and evaluation | Raw baseline at 150 ms | Split file and working TCNN | Baseline accuracy, kappa, F1 reported | Models saved at MAKE Archive (paths); metric plots, training curves, model size |
| 2.4 Preprocessing | Measure the effect of the notch and the filter grid | Baseline available | Retained preprocessing chain chosen | Same as 2.3, plus ICA maps and source analysis |
| 2.5 Optimisation | Reduce the model for the board | Retained preprocessing chain | Reduced model meets REQ-SW-004/005 | Same as 2.3 for each reduction step; window-size grid and robustness score |
| 2.6 Science | Interpret the CNN filters and test architecture changes | Trained TCNN available (runs in parallel with 2.5) | Hypotheses confirmed or rejected by ablation | Structured docs base and literature database |

### 3.3 Functionally linked systems

- **In-House EMG Acquisition Hardware**: source of live EMG; defines ADC resolution, sampling rate and the analogue 20–500 Hz first-order filter that the dataset adaptation mimics.
- **Embedded firmware** ([BMI-EMG-Firmware](https://github.com/N-Pulse/BMI-EMG-Firmware/tree/i2c_adc)): will host the reduced model from step 2.5 and its peripherics.
- **DAQ team and in-house recordings (MAKE Archive)**: the in-house recordings, acquired with our board and our own gesture set, are the target training data for the decoder. In V2 they are used to measure the domain gap with the adapted EPN612 data; as soon as enough sessions are available they will be used to fine-tune the EPN-trained model and, eventually, to train on our data directly. The ingestion path (XDF → BIDS → loaders) already exists for this (REQ-SW-001). The archive also stores the trained models.
- **EMG-EPN-612 dataset (Zenodo)**: training data for V2.
- **V1 decoder** (feature-based TreeOfTrees): the baseline V2 must outperform (REQ-SW-011).
- **EEG decoding** ([BMI-SOFT-EEG-Decoding](https://github.com/N-Pulse/BMI-SOFT-EEG-Decoding)): parallel modality; may reuse the TCNN and the reduction pipeline. May be used down the line with EMG and some sort of state machine.

### 3.4 Performance budgets and design margins

| Budget | Target | Current estimate / value | Source |
|---|---|---|---|
| Inference latency | < 200 ms (REQ-SW-005) | Not measured | Step 2.5 |
| Window size / step | Realistic window (25–200 ms tested) | 150 ms window, 25 ms step until step 2.5; then window grid 50–200 ms × step size | V2 plan |
| Flash | ≤ 2048 KB | Not measured; reduced by channel silencing and quantisation | STM32WBA65RI |
| SRAM | ≤ 512 KB | Circular buffer holding one window, 8 ch × 300 samples (150 ms at 2 kHz, 16-bit) ≈ 4.8 KB; with a 25 ms step only the newest 50 samples per channel are written before each inference, so overlap adds no buffer cost; activations TBD | derived |

> Budgets will be filled with measured values at step 2.5 (OI-05).

---

## 4. Traded-off concepts

### 4.1 Trade-off method

The trade-off method used was based on the following criteria :
- Meets performance requirements (accuracy, kappa, F1 on the subject-disjoint split)
- Embedded feasibility (memory, compute, latency on STM32WBA65RI after reduction)
- Potential to outperform V1 and to be improved (REQ-SW-011, science track)
- Data efficiency (works with adapted public data)
- Development effort and tooling maturity
- Maintainability


| Criterion | Weight (1 - 5) | Rationale for weight |
|---|---|---|
| Meets performance requirements | 5 | The decoder must reliably separate the 6 gestures across subjects. |
| Embedded feasibility | 4 | The model must run on the board this semester, but the reduction pipeline gives room to shrink it. |
| Potential to outperform V1 | 4 | V2 exists to beat V1 and to open a research path. |
| Data efficiency | 3 | EPN612 provides 612 users, so data is less of a constraint than in V1. |
| Development effort | 3 | Student team with limited time per semester. |
| Maintainability | 2 | Useful for handover between terms, not critical at PoC stage. |

Score scale :
> [1] : Does not meet any requirements
> [2] : Meets some requirements
> [3] : Meets all minimum requirements 'Shall' requirements
> [4] : Meets all minimum requirements and 'should'
> [5] : Meets all requirements, 'Shall, should & may'

> Scores are qualitative at this stage and will be updated with measured results from steps 2.3–2.5 (OI-06). Max score = 5 × (5+4+4+3+3+2) = 105.

### 4.2 Traded-off design choices

#### 4.2.1 Alternative A: V1 feature-based TreeOfTrees

- **Description:** Hand-crafted time and frequency features (24 types × 8 channels) fed to a binary decision tree that differentiates between movement 1 and not movement 1. If it decodes not movement 1 then it would go to the movement 2 binary tree and so on.
- **Pros:** Small, fast to train, interpretable, already implemented and levereged easy to decode movements.
- **Cons:** Performance capped by feature quality; feature extraction dominates MCU compute; little room for research improvement. Default DT and SVM kept as the V1 baseline to beat.

| Criterion | Weight (1-5) | score (1-5) |
|---|---|---|
| Meets performance requirements | 5 | 3 |
| Embedded feasibility | 4 | 4 |
| Potential to outperform V1 | 4 | 1 |
| Data efficiency | 3 | 4 |
| Development effort | 3 | 5 |
| Maintainability | 2 | 4 |
| ---|---|---|
|Total Score| 70 | max score : 105 |

#### 4.2.2 Alternative B: Remodelled / larger architectures

- **Description:** More complex CNN or architectures that worked with attention were discussed
- **Pros:** Closer to what SOTA is going for, with chances of increased in classification metrics
- **Cons:** V2 is our first deep learning model, going for a complex architecture would lock us with a copy from literature that we wouldnt have the intuiton to upgrade. It would be harder to make sure it would fit on the lower compute. Not just that, but training compute and data availability could be an issue. CNN allows for easier training with an intuitive architecture and chnaces of filter intepretability.

| Criterion | Weight (1-5) | score (1-5) |
|---|---|---|
| Meets performance requirements | 5 | 4 |
| Embedded feasibility | 4 | 1 |
| Potential to outperform V1 | 4 | 3 |
| Data efficiency | 3 | 2 |
| Development effort | 3 | 2 |
| Maintainability | 2 | 2 |
| ---|---|---|
|Total Score| 52 | max score : 105 |

#### 4.2.3 Retained: Temporal CNN (Atzori et al. 2016) + Wang et al. (2020) reduction

- **Pros:** Simple, EMG-specific, reproducible from the paper; small enough to reduce for the board; first and last layers open to interpretation.
- **Cons:** Less interpretable than V1; adds PyTorch; baseline performance expected to be modest before preprocessing and fine tuning.

| Criterion | Weight (1-5) | score (1-5) |
|---|---|---|
| Meets performance requirements | 5 | 4 |
| Embedded feasibility | 4 | 4 |
| Potential to outperform V1 | 4 | 4 |
| Data efficiency | 3 | 3 |
| Development effort | 3 | 4 |
| Maintainability | 2 | 4 |
| ---|---|---|
|Total Score| 81 | max score : 105 |

### 4.3 Trade-off summary

| Option | Score | Retained | Key reason |
|---|---|---|---|
| Temporal CNN + Wang reduction | 81 | Yes | Simple, reducible, interpretable filters, room to upgrade |
| V1 feature-based TreeOfTrees | 70 | No (baseline) | Reference V2 must outperform |
| Remodelled / larger architectures | 52 | No (science track) | Too heavy for the board and too early for our first deep model |

## 5. Quality management and compliance

**WORK IN PROGRESS**

Current status: research-use only (RUO) proof of concept; no clinical use. Good-practice measures in place: pytest suite (`tests/`), pre-commit, black/pylint, PR and issue templates, reviewed merges into `main`.

<!--
### 5.1 Regulatory classification and pathway

| Item | Statement |
|---|---|
| Intended use classification | `<RUO / IVD / medical device / bioprocess / environmental / other>` |
| Target jurisdictions | `<EU / US / CH / UK / other>` |
| Applicable regulation | `<e.g. EU IVDR 2017/746, EU MDR 2017/745, FDA 21 CFR 820 (QMSR), 21 CFR Part 11>` |
| Device / product class | `<class and rationale>` |
| Regulatory strategy | `<summary, notified body / agency contact status>` |

### 5.2 Quality management system

- **QMS framework:** `<ISO 13485 / ISO 9001 / GLP / GMP / GCP / lightweight in-house QMS>`
- **QMS scope and document hierarchy:** `<link>`
- **Design controls followed:** `<design inputs → outputs → review → verification → validation → transfer → changes>`
- **This document's place in the design history file (DHF):** `<link>`

### 5.3 Applicable standards and guidelines

| Standard / Guideline | Title | Applicability | Compliance status |
|---|---|---|---|
| ISO 13485 | Medical devices: Quality management systems | | Planned / Partial / Compliant / N/A |
| ISO 14971 | Application of risk management to medical devices | | |
| IEC 62304 | Medical device software life cycle | | |
| IEC 61010-1 | Safety of electrical equipment for measurement, control, laboratory use | | |
| IEC 62366-1 | Usability engineering | | |
| ISO 20387 | Biobanking: general requirements | | |
| `<other>` | | | |


### 5.4 Ethics, access, and biosecurity compliance

| Topic | Applicability | Status / reference |
|---|---|---|
| Ethics committee / IRB approval | | |
| Informed consent | | |
| Nagoya Protocol / access and benefit sharing | | |
| Biosafety committee approval | | |
| Import / export / dual-use controls | | |
| Animal research regulations | | |

-->

## 6. Licensing

**WORK IN PROGRESS**

<!--
| Topic | Decision |
|---|---|
| Hardware license | `<CERN-OHL-S / CERN-OHL-W / TAPR / other>` |
| Software license | `<Apache-2.0 / GPL-3.0 / MIT / other>` |
| Documentation and data license | `<CC-BY-4.0 / CC0 / other>` |
| Open Source Hardware certification | `<OSHWA status>` |
| Licence compatibility of all dependencies | `<checked on YYYY-MM-DD, link to SBOM>` |
| Patents / IP / prior art | `<statement, defensive publication?>` |
| Contribution and governance model | `<link to CONTRIBUTING.md / GOVERNANCE.md>` |
| Liability and warranty disclaimer | `<statement, especially for any clinical use>` |
| Sourcing for builders | `<BOM with alternates, availability of parts worldwide>` |
| Quality of community-contributed changes | `<review gates, required tests, who can merge>` |

-->


## 7. Open issues and future work

| ID | Issue / TBD / TBC | Owner | Due | Impact if unresolved |
|---|---|---|---|---|
| OI-04 | Quantify signal similarity between adapted EPN612 and local recordings; produce electrode maps | EMG team | TBD | Domain gap unknown |
| OI-05 | Measure latency, Flash and RAM of the reduced model on STM32WBA65RI | EMG team / Firmware | TBD | REQ-SW-004/005 unverified |
| OI-06 | Update chapter 4 scores with measured accuracy, kappa and F1 | EMG team | TBD | Trade-off stays qualitative |
| OI-07 | Define the window/step robustness score used in step 2.5 | EMG team | TBD | Window choice not justified |
| OI-08 | Set up the structured docs base and literature database for the science track | EMG team | TBD | Findings lost between terms |

---

## 8. References

1. Applicable documents: project requirements specification `<TBD>`; [`docs/adapt_to_board.md`](adapt_to_board.md); [`docs/diagrams/class.puml`](diagrams/class.puml); [`src/bmiemg/download/README.md`](../src/bmiemg/download/README.md).
2. Reference documents:
   - Atzori, M., Cognolato, M., & Müller, H. (2016). Deep learning with convolutional neural networks applied to electromyography data: A resource for the classification of movements for prosthetic hands. *Frontiers in Neurorobotics*, 10. https://doi.org/10.3389/fnbot.2016.00009
   - Wang, X., Hersche, M., Tömekce, B., Kaya, B., Magno, M., & Benini, L. (2020). An accurate EEGNet-based motor-imagery brain-computer interface for low-power edge computing. arXiv:2004.00077. https://doi.org/10.1109/MeMeA49120.2020.9137134
   - Liao et al. (2025). `<full reference TBD>`
   - Benalcázar M. et al., *EMG-EPN-612 dataset*, Zenodo.
   - STMicroelectronics, NUCLEO-WBA65RI / STM32WBA65RI datasheet.
   - PyTorch documentation.

---

## 9. Appendices

### Appendix A: Bill of materials

Software-only subsystem: no hardware BOM. Dependencies are pinned in [`pyproject.toml`](../pyproject.toml) and [`poetry.lock`](../poetry.lock) (Python 3.12).
