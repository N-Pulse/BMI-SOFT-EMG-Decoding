# Adapting the Model to an Embedded System - Research & Assessment

**Project:** N-pulse Prosthetic arm control using Machine Learning on EMG (and EEG) signals  
**Goal:** Prepare the integration pipeline to deploy an ML model on an embedded system

---

## 1. Context & Problem Statement

The N-pulse project aims to control a prosthetic arm in real time by recognizing hand gestures from electromyography (EMG) signals and EEG signals. We need to prepare the embedded integration pipeline for future deployment once the model is ready. In this work, we will focus on EMG signals.

The core challenge is to run ML inference directly on a microcontroller, under strict constraints of memory, compute, and latency. By ML inference, we mean the process by which the trained model receives new data and produces a prediction: the microcontroller receives a short window of EMG samples, feeds them through the model, and outputs a gesture class (e.g. open hand, pinch, fist) in real time, entirely on-device, with no connection to a PC or cloud server.

---

## 2. Target Hardware: NUCLEO-F334R8

The board available for the n-pulse project is the **STMicroelectronics NUCLEO-F334R8**, based on the STM32F334R8 microcontroller. Its key specifications are:

| Parameter | Value |
|---|---|
| Core | ARM Cortex-M4 with FPU |
| Clock speed | 72 MHz |
| Flash memory | 64 KB |
| SRAM | 12 KB |
| Toolchain | STM32CubeIDE |

This board was selected as it is already in use within the project team (shared with the EEG signal pipeline). It is a low-cost, low-power development board typical of early-stage embedded research.


---

## 3. Literature Review: Analysis of Reference Papers

### Paper 1: On the Deployment of Edge AI Models for Surface Electromyography-Based Hand Gesture Recognition

**Signal and data:** Surface EMG from 20 healthy subjects, 8 channels at 500 Hz, 6 hand gestures (pinch variants, closed fist, rest), very similar setup to n-pulse.

**Models deployed:** Bagged Forest (100 trees) and a shallow Neural Network (1 hidden layer, 144 ReLU neurons), both on an ESP32-S3 (240 MHz dual-core).

**Key findings relevant to n-pulse:**

- Reducing features from 144 to 80 or 64 channel-specific features produced up to a **31% reduction in total pipeline execution time** with negligible accuracy loss.
- The main computational bottleneck was **feature extraction**, not model inference , specifically FFT-based frequency-domain features carry O(n³) complexity, whereas time-domain features (RMS, MAV, WL, ZC) carry O(n) complexity. RMS from channel 4 alone accounted for over 25% of Bagged Forest accuracy.
- The interrupt-driven firmware architecture (500 Hz base interrupt, circular buffer, 200 ms segmentation windows, modular filtering and inference blocks) is the correct way to build any real-time biosignal pipeline and is **directly reusable for n-pulse**.
- Statistical validation methodology (1008 on-device predictions at 95% confidence) confirmed negligible accuracy loss from PC-to-MCU model conversion , this approach is reusable for n-pulse.

**Result:** Neural Network with 80 best features achieves 94.21% accuracy on the embedded system, 4.66 ms inference time, 449 KB memory footprint.

### Paper 2: Microcontroller Implementation of LSTM Neural Networks for Dynamic Hand Gesture Recognition

**Signal and data:** 3-axis accelerometer from a wrist-worn smartwatch at ~9.1 Hz (SmartWatch Gesture Dataset). This is a different signal modality from n-pulse , the main value of this paper is its **deployment methodology**, not its signal pipeline.

**Model deployed:** Two stacked LSTM layers (100 + 50 units) + two dense layers, on an STM32L476RG (80 MHz Cortex-M4, 1 MB Flash, 128 KB SRAM).

**Key findings relevant to n-pulse:**

- **Analytical memory sizing before training:** the authors derived closed-form expressions for weight count and MACs as a function of layer dimensions, guaranteeing the model fits the target hardware before any training is done. This approach is reusable for any architecture.
- **Data augmentation** (noise addition 5% σ, amplitude scaling ±10%, time warping ±30%) was used to double the training set size , directly applicable to n-pulse where collecting large labeled EMG datasets from many subjects is expensive.
- **Subject-based cross-validation** (4-fold) ensures evaluation on unseen subjects , a non-negotiable validation requirement for a prosthetic that must generalize to its user.
- Paper 2's Table 5 shows that a Cortex-M7 class board (e.g. STM32H743, 480 MHz) reduces LSTM inference from 418 ms to ~37–51 ms, comfortably within the 200 ms budget.

**Result:** 90.25% accuracy (4-fold subject-based cross-validation), 292 KB Flash, 17 KB RAM, 418 ms inference on an 80 MHz Cortex-M4.

---

## 4. Constraints Analysis for the NUCLEO-F334R8

### 4.1 Memory

The NUCLEO-F334R8 provides only **64 KB of Flash** and **12 KB of RAM**. This is extremely limited compared to what the reference literature requires:

- Paper 1's Bagged Forest (100 trees): **530 KB** , 8× more than available Flash
- Paper 1's Neural Network (80 features): **449 KB** , still impossible
- Paper 2's LSTM: **292 KB Flash + 17 KB RAM** , RAM alone exceeds the board's 12 KB

This means that the models tested in both reference papers **cannot be deployed on this board as-is**. Memory is the primary and most critical constraint for the NUCLEO-F334R8.

### 4.2 Compute

At 72 MHz, the Cortex-M4 core from our current board is slightly slower than the 80 MHz board used in Paper 2, which already achieved 418 ms inference for an LSTM, far exceeding the real-time budget. For complex models, compute is therefore also a concern on this board.

However, for simpler models such as logistic regression (see Section 5), the compute cost is trivial: inference reduces to a single matrix multiplication over the feature vector, which completes in microseconds at 72 MHz. Compute is therefore not a bottleneck for lightweight classifiers.

### 4.3 Latency

For a prosthetic arm, the delay between muscle contraction and gesture classification must be imperceptible to the user. A total pipeline latency above 200–300 ms breaks the natural feel of the device. The full pipeline includes signal acquisition, preprocessing (filtering + feature extraction), model inference, and output.

Based on Paper 1, feature extraction is the dominant cost , even on a 240 MHz board, it takes approximately 10 ms. On the 72 MHz NUCLEO-F334R8, this is estimated to scale to 30–40 ms. For a lightweight classifier such as logistic regression, inference adds negligible time, meaning the **200 ms latency target is achievable** if time-domain features are used. For complex models (Random Forest, LSTM), latency would likely exceed the budget.


---

## 5. Assessment for n-pulse: Current Model Compatibility and Solutions

### 5.1 Current Model: Random Forest

The current n-pulse model is a **Random Forest**, a classical ML classifier. Unlike neural networks, it does not require TFLite or STM32Cube.AI for deployment , it can be converted to optimized C code using tools such as **micromlgen** or **Everywhereml**, generating a series of if-else decision tree statements that run directly on the microcontroller.

However, the memory requirements of a standard Random Forest (100 trees) are incompatible with the NUCLEO-F334R8. Even after aggressive pruning (reduced tree count, limited depth) and feature selection, the gap between what the literature requires (~512 KB minimum) and what the board offers (64 KB Flash, 12 KB RAM) is too large to bridge without unacceptable accuracy loss.

**Feature selection alone is insufficient:** going from 144 to 64 features reduced Bagged Forest memory from 530 KB to 512 KB in Paper 1 , a 3.5% improvement, far from the 8× reduction needed.

### 5.2 Alternative Solution: Logistic Regression

Given the board's constraints, **logistic regression** is the most viable model candidate. As a linear classifier, it stores only one coefficient per input feature plus a bias term per class. For 80 features and 6 gesture classes, this amounts to approximately:

```
80 features × 6 classes × 4 bytes (float32) = ~1.9 KB
```

This fits comfortably within the board's 64 KB Flash, using less than 3% of available memory. Inference is also trivially fast , a single matrix multiplication over the feature vector , meaning the 200 ms latency target is met with ease.

The open question for logistic regression is **accuracy**: EMG gesture recognition is a non-linear problem, and linear classifiers may not achieve the 90%+ accuracy seen with Random Forest or Neural Networks in the literature. This trade-off between hardware feasibility and model expressiveness is the key design decision for n-pulse and should be validated experimentally once data is available.

### 5.3 Alternative Solution: Board Upgrade

If model accuracy is prioritized over hardware cost and power consumption, migrating to a more capable board would substantially simplify the deployment pipeline:

| Board | Core | Clock | Flash | RAM | Feasibility |
|---|---|---|---|---|---|
| NUCLEO-F334R8 (current) | Cortex-M4 | 72 MHz | 64 KB | 12 KB | Only logistic regression |
| STM32L476RG (Paper 2) | Cortex-M4 | 80 MHz | 1 MB | 128 KB | LSTM feasible |
| STM32H743 | Cortex-M7 | 480 MHz | 2 MB | 1 MB | Full pipeline, 37 ms inference |

A Cortex-M7 class board would support Random Forest, Neural Networks, and even LSTM models with inference well within the 200 ms budget. This comes at the cost of higher price (~€50–100 vs ~€15) and higher power consumption, which must be weighed against the wearable nature of a prosthetic device.

This trade-off should be evaluated by the project team before committing to a deployment architecture.

---

## 6. No Longer Relevant from the Papers

Given that the current model is a **Random Forest** (not a neural network) and the target board is the **NUCLEO-F334R8**, the following elements from the reference papers do not apply to n-pulse in its current state:

- **Keras → TFLite → STM32Cube.AI pipeline (Paper 2):** this conversion chain is specific to neural networks. For a Random Forest or logistic regression, the equivalent is sklearn → micromlgen/Everywhereml → C code → STM32CubeIDE.
- **Analytical LSTM memory sizing formulas (Paper 2):** the closed-form expressions for LSTM weight counts and MACs are architecture-specific and do not apply to tree-based or linear models.
- **BatchNormalization, Dropout, softmax (Paper 2):** these are neural network training and regularization concepts with no equivalent in Random Forest or logistic regression.
- **The ESP32-S3 as deployment target (Paper 1):** while validated as a capable platform, it is not the board in use at n-pulse. Its specs (240 MHz, sufficient Flash) are referenced only for comparison.

---

## 7. Deployment Pipeline (Recommended)

Based on the above analysis, the recommended pipeline for n-pulse given current constraints is:

```
Train model in sklearn (Python)
        ↓
Feature selection (time-domain features only: RMS, MAV, WL, ZC)
        ↓
Convert to C code using micromlgen / Everywhereml
        ↓
Integrate into STM32CubeIDE firmware
        ↓
Interrupt-driven real-time pipeline on NUCLEO-F334R8
        ↓
On-device validation (statistical sampling at 95% confidence)
```

If the board is upgraded to a neural-network-compatible platform, the pipeline becomes:

```
Train model in Keras (Google Colab)
        ↓
Convert to TFLite
        ↓
Pass through STM32Cube.AI (X-CUBE-AI) → optimized C code
        ↓
Flash to STM32 board via STM32CubeIDE
        ↓
On-device validation
```

---

## 8. Open Questions & Next Steps

- **Validate logistic regression accuracy** on n-pulse EMG data once available , determine if it meets project requirements.
- **Decide on board upgrade** , weigh accuracy needs against power and cost constraints for a wearable prosthetic.
- **Implement interrupt-driven firmware pipeline** (reusable from Paper 1 regardless of model choice).
- **Apply cross-validation** during training to ensure model generalization across users.

