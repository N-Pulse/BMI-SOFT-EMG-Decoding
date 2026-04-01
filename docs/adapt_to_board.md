# Adapting the Model to an Embedded System - Research & Assessment

**Project:** N-pulse Prosthetic arm control using Machine Learning on EMG signals  
**Goal:** Prepare the integration pipeline to deploy an ML model on an embedded system

---

## 1. Context & Problem Statement

The n-pulse project aims to control a prosthetic arm in real time by recognizing hand gestures from electromyography (EMG) signals and EEG signals. We need to prepare the embedded integration pipeline for future deployment once the model is ready. In this work, we will focus on EMG signals.

The core challenge is to run ML inference directly on a microcontroller, under strict constraints of memory, compute, and latency. By ML inference, we mean the process by which the trained model receives new data and produces a prediction: the microcontroller receives a short window of EMG samples, feeds them through the model, and outputs a gesture class (open hand, pinch, fist) in real time, entirely on-device, with no connection to a PC or cloud server.

- **Memory:** The model weights and intermediate activations must fit within the microcontroller's available RAM and Flash. For example, a typical embedded offers only a range of 1 MiB to 4MiB of total memory, which severely limits model depth and width. Every layer added to the network has a direct cost in memory. This must be accounted for at design time.

- **Compute:** Microcontrollers operate at low clock speeds (typically 80 - 240 MHz) and have no GPU or dedicated AI accelerator. Every arithmetic operation in the forward pass multiplications, additions, activation functions must complete within the available CPU cycles. Complex architectures like deep LSTMs or transformers are likely too expensive without aggressive optimization. We’ll prioritize time domain feature.

- **Latency:** For a prosthetic arm, the delay between muscle contraction and gesture classification must be imperceptible to the user. A latency above ~200 - 300 ms will break the natural feel of the device. This means the entire pipeline signal acquisition, preprocessing, inference, and output must complete well within that window.


---

## 2. Target Hardware: NUCLEO-WBA65RI

The board available for the n-pulse project is the **STMicroelectronics NUCLEO-WBA65RI**, based on the STM32WBA65RI microcontroller. Its key specifications are:

| Parameter | Value |
|---|---|
| Core | ARM Cortex-M33 with FPU |
| Clock speed | 100 MHz |
| Flash memory | 2048 KB (2 MB) |
| SRAM | 512 KB |
| Toolchain | STM32CubeIDE |

This board provides excellent resource availability for embedded ML !

---

## 3. Literature Review: Analysis of Reference Papers

### Paper 1: On the Deployment of Edge AI Models for Surface Electromyography-Based Hand Gesture Recognition

**Signal and data:** Surface EMG from 20 healthy subjects, 8 channels at 500 Hz, 6 hand gestures (pinch variants, closed fist, rest), very similar setup to n-pulse.

**Models deployed:** Bagged Forest (100 trees) and a shallow Neural Network (1 hidden layer, 144 ReLU neurons), both on an ESP32-S3 (240 MHz dual-core).

**Key findings relevant to n-pulse:**

- Reducing features from 144 to 80 or 64 channel-specific features produced up to a **31% reduction in total pipeline execution time** with negligible accuracy loss.
- The main computational bottleneck was **feature extraction**, not model inference. Specifically, FFT-based frequency-domain features carry O(n³) complexity, whereas time-domain features (RMS, MAV) carry O(n) complexity. RMS from channel 4 alone accounted for over 25% of Bagged Forest accuracy.
- The interrupt-driven firmware architecture (500 Hz base interrupt, circular buffer, 200 ms segmentation windows, modular filtering and inference blocks) is the correct way to build any real-time biosignal pipeline and is **directly reusable for n-pulse**.
- Statistical validation methodology (they did 1008 on-device predictions at 95% confidence) confirmed negligible accuracy loss from PC-to-MCU model conversion. This approach is reusable for n-pulse: we can compare accuracy on PC (original model) vs MCU (deployed C code) using the same test dataset to ensure the conversion does not degrade performance

**Result:** Neural Network with 80 best features achieves 94.21% accuracy on the embedded system, 4.66 ms inference time, 449 KB memory footprint.

### Paper 2: Microcontroller Implementation of LSTM Neural Networks for Dynamic Hand Gesture Recognition

**Signal and data:** 3-axis accelerometer from a wrist-worn smartwatch at ~9.1 Hz (SmartWatch Gesture Dataset). This is a different signal modality from n-pulse. The main value of this paper is its **deployment methodology**, not its signal pipeline.

**Model deployed:** Two stacked LSTM layers (100 + 50 units) + two dense layers, on an STM32L476RG (80 MHz Cortex-M4, 1 MB Flash, 128 KB SRAM).

**Key findings relevant to n-pulse:**

- **Analytical memory sizing before training:** the authors derived closed-form expressions for weight count and MACs as a function of layer dimensions, guaranteeing the model fits the target hardware before any training is done. This approach is reusable for any architecture.
- **Data augmentation** (noise addition 5% σ, amplitude scaling ±10%, time warping ±30%) was used to double the training set size. This is directly applicable to n-pulse where collecting large labeled EMG datasets from many subjects is expensive.
- **Subject-based cross-validation** (4-fold) ensures evaluation on unseen subjects. This is a non-negotiable validation requirement for a prosthetic that must generalize to its user.

**Result:** 90.25% accuracy (4-fold subject-based cross-validation), 292 KB Flash, 17 KB RAM, 418 ms inference on an 80 MHz Cortex-M4.

---

## 4. Constraints Analysis for the NUCLEO-WBA65RI

### 4.1 Memory

The NUCLEO-WBA65RI provides **2048 KB (2 MB) of Flash** and **512 KB of RAM**. This is a substantial resource pool for embedded ML deployment:

**Model feasibility:**
- Paper 1's Bagged Forest (100 trees): **530 KB** →  **Fits easily** (26% of available Flash)
- Paper 1's Neural Network (80 features): **449 KB** →  **Fits easily** (22% of available Flash)
- Paper 2's LSTM: **292 KB Flash + 17 KB RAM** →  **Fits easily** (14% Flash, 3% RAM)

**Conclusion:** Memory is **not a bottleneck** for n-pulse. The WBA65RI can be used with Random Forest, Neural Networks, and even LSTM models, all feasible deployment targets.

### 4.2 Compute

At 100 MHz, the Cortex-M33 core is sufficient for real-time gesture recognition. Feature extraction dominates computational complexity: time-domain features (RMS, MAV, ZC) require O(n) operations per sample, while frequency-domain features (FFT) require O(n³) complexity (Paper 1). By prioritizing time-domain features, the total computational load for the complete pipeline (feature extraction + Random Forest inference) remains modest compared to the available CPU cycles.

### 4.3 Latency

For a prosthetic arm, the delay between muscle contraction and gesture classification must be imperceptible to the user. A total pipeline latency above 200-300 ms breaks the natural feel of the device. The full pipeline includes signal acquisition, preprocessing (filtering + feature extraction), model inference, and output.

Based on Paper 1, feature extraction dominates at ~10 ms on a 240 MHz system. On the 100 MHz WBA65RI, this is estimated at **20-25 ms**. Random Forest inference (even 100 trees) adds minimal overhead (~5-10 ms). The **200 ms latency target is easily achievable** on this hardware.

---

## 5. Assessment for n-pulse: Model Compatibility and Recommendations

### 5.1 Primary Recommendation: Random Forest (Unmodified)

The current **Random Forest model (100 trees) is fully viable** on the NUCLEO-WBA65RI:

- **Memory:** ~530 KB → uses 26% of available Flash, leaves ample room for firmware, and future features
- **Inference latency:** estimated 5-10 ms (Paper 1's baseline), well within the 200 ms budget
- **Deployment pipeline:** sklearn → micromlgen/Everywhereml → C code → STM32CubeIDE

**This is the recommended path forward.** The team can deploy the current Random Forest model without retraining or modification.

### 5.2 Secondary Option: Neural Network (if higher accuracy desired)

If Random Forest accuracy is insufficient once data is collected, upgrading to the **shallow neural network from Paper 1** is straightforward:

- **Model:** 1 hidden layer, 144 ReLU neurons, 80 input features
- **Memory footprint:** ~449 KB (22% of available Flash)
- **Inference latency:** 4.66 ms (Paper 1)
- **Accuracy:** 94.21% on the same EMG gesture task
- **Deployment pipeline:** Keras → TFLite → STM32Cube.AI (X-CUBE-AI) → optimized C code → STM32CubeIDE

This option is viable if higher accuracy is needed and the current Random Forest does not meet performance targets.

---

## 6. Deployment Pipeline 

Recommended pipeline for n-pulse on the NUCLEO-WBA65RI is:

```
Train Random Forest model in sklearn (Python)
        ↓
Feature selection (time-domain features: RMS, MAV, WL, ZC)
        ↓
Convert to C code using micromlgen / Everywhereml
        ↓
Integrate into STM32CubeIDE firmware
        ↓
Interrupt-driven real-time pipeline on NUCLEO-WBA65RI
        ↓
On-device validation (statistical sampling at 95% confidence)
```

**Alternative pipeline from paper 2(if upgrading to neural network):**

```
Train model in Keras (Google Colab)
        ↓
Convert to TFLite
        ↓
Pass through STM32Cube.AI (X-CUBE-AI) → optimized C code
        ↓
Flash to NUCLEO-WBA65RI via STM32CubeIDE
        ↓
On-device validation
```

Both pipelines are straightforward and require no custom firmware architecture. The interrupt-driven signal processing pipeline from Paper 1 remains the correct approach regardless of model choice.

---

## 7. Next Steps

- **Implement (convert code in C) and test Random Forest deployment** on the WBA65RI using micromlgen → verify actual inference latency and energy consumption on hardware.
- **subject-based cross-validation** to ensure generalization across users.
- **Validate Random Forest accuracy**, if insufficient, potentially evaluate other ml model upgrade path
- **Interrupt-driven firmware pipeline (writing code on MCU)** (following Paper 1): 500 Hz acquisition, circular buffer, 200 ms segmentation windows, modular filtering and inference.
- **Statistical validation** following Paper 1's methodology to confirm negligible accuracy loss in PC-to-MCU conversion.

---

## 8. Conclusion

The NUCLEO-WBA65RI provides excellent resource availability for embedded ML deployment of the n-pulse gesture recognition pipeline. The team can deploy the current Random Forest model without modification, or invest in higher-accuracy architectures (neural networks, LSTM) with confidence that the hardware can support them. To optimize latency, from paper 1, time-domain features should be used the most possible instead of frequency-domain features (FFT), as they reduce feature extraction complexity from O(n³) to O(n). According to the paper 1, it achieved ~31% pipeline speedup while maintaining accuracy. From paper 1 and 2 computed memory usage, NUCLEO-WBA65RI can deploy all practical models for EMG/gesture recognition (Random Forest, Neural Network, LSTM). For models like large-scale Vision Transformers or uncompressed deep neural networks (>10 MB), deployment would not be feasible (not applicable to our project). Latency and inference speed are the limiting factorsbut they are well within the 200 ms real-time budget for prosthetic arm control, with an estimated 20-25 ms feature extraction and <10 ms model inference on the WBA65RI (based on comparison with paper 1 using bagged forest)

