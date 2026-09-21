# Chest X-Ray Pneumonia Detection: CNN vs. RNN

A deep learning course project (Deep Learning 466, SUNY New Paltz) that asks whether CNN and RNN models can accurately detect pneumonia from chest X-ray images, and how the two architectures compare. It builds custom models in PyTorch, trains them on the same data, and evaluates them against a naive majority-class baseline.

**Team:** Rashad Ricketts and Cole Potrzeba. The full presentation is in [`docs/`](docs/).

## Motivation

Most published work on pneumonia detection from chest X-rays uses CNNs, with reported accuracies ranging from about 74% to 98% depending on architecture and augmentation strategy. A few experimental papers have also treated image pixels as sequential data for RNNs, which looks promising for compact models. We wanted to build our own architectures and compare a CNN against an RNN directly on the same dataset, to see for ourselves why architecture matters for image data. Medical imaging made a good test case because the cost of a wrong prediction is clear.

## Data & Methods

- **Source:** The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset on Kaggle, originally published by Kermany et al. (2018). The images are pediatric chest X-rays from patients aged 1 to 5 at Guangzhou Women and Children's Medical Center, and the dataset is widely used as a benchmark.
- **Size and balance:** 5,216 training images labeled `NORMAL` (1,341) or `PNEUMONIA`, roughly 74% pneumonia and 26% normal. The test set has 624 images.
- **Preprocessing:** Images were resized to 128×128 and converted to grayscale for the final CNN models, and class imbalance was handled with a class-weighted cross-entropy loss.
- **CNN models:** Three iterations, each adding to the last.
  - *Model 1:* 5×5 kernels, three convolutional layers, two fully connected layers, ReLU and max pooling.
  - *Model 2:* 3×3 kernels, four convolutional layers, two fully connected layers, ReLU, max pooling, and batch normalization.
  - *Model 3:* Model 2 plus an extra fully connected layer and Leaky ReLU in place of ReLU.
- **RNN model:** Each image is reshaped into a sequence of pixel rows and fed to a recurrent network, which is a workaround since RNNs are designed for sequence data rather than images.
- **Baseline:** A naive model that always predicts the majority class, used as the reference point for every other model.
- **Evaluation:** Accuracy on the held-out test set, plus recall, specificity, precision, and F1 for the best CNN.

| Model                                   | Test Accuracy |
| --------------------------------------- | ------------- |
| Naive baseline (always predicts majority) | 62.5%       |
| RNN                                     | 62.5%         |
| Simple CNN (baseline in the notebook)   | 80.5%         |
| **CNN Model 2 (best)**                  | **84.78%**    |

| CNN Model 2 Metric   | Value |
| -------------------- | ----- |
| Recall (Sensitivity) | 0.913 |
| Specificity          | 0.739 |
| Precision (PPV)      | 0.854 |
| F1 Score             | 0.883 |

## Key Findings

- **The best CNN beat the naive baseline by about 22 percentage points** (84.78% vs. 62.5%), showing it learned real patterns in the X-rays rather than just the class ratio.
- **The RNN did no better than the naive baseline.** Its 62.5% accuracy matches always guessing pneumonia. Reshaping an image into a sequence of pixel rows breaks the 2D spatial relationships that matter for image understanding, so a CNN has a built-in advantage on this kind of data.
- **High recall matters most for screening.** Model 2's recall of 0.913 means it catches about 91% of actual pneumonia cases. A missed diagnosis (false negative) carries more clinical risk than an unnecessary follow-up.
- **Precision shows the trade-off.** A precision of 0.854 means about 15% of positive predictions are false alarms, which is acceptable in a decision-support role where a physician reviews every flagged case.
- **Results fit the literature.** At 84.78% accuracy and 0.883 F1, Model 2 is consistent with simpler single-CNN models in published work (roughly 74% to 90%) and below ensemble and attention-based CNNs (around 94% accuracy, F1 above 90%), which is what we would expect from an introductory custom model.

## Limitations

- **No clinical validation:** The model has not been tested prospectively or in a clinical setting and should not be used for real diagnosis.
- **Narrow population:** All images come from pediatric patients (ages 1 to 5) at a single medical center, so the results may not generalize to adults or other hospitals and imaging equipment.
- **Lower specificity:** A specificity of 0.739 means a meaningful share of healthy patients would be flagged incorrectly.
- **Small validation set:** The dataset's validation folder has only 16 images, so validation metrics were noisy and not a reliable guide for model selection.
- **Limited tuning:** The models were trained from scratch with no extensive hyperparameter tuning, and no pretrained networks were used.
- **RNN is a poor fit for images:** The RNN comparison shows the mismatch between the architecture and the data as much as it compares the two model types on equal terms.

## Future Work

Hyperparameter tuning, transfer learning from pretrained networks such as ResNet or VGG, larger and more diverse datasets, and prospective clinical validation before any real-world use.

## Skills Demonstrated

Image classification · convolutional and recurrent neural network design in PyTorch · handling class imbalance with weighted loss · model iteration and comparison against a baseline · evaluation beyond accuracy (recall, specificity, precision, F1) · connecting model results to real-world impact · literature review

## Tools & Stack

Python · PyTorch · torchvision · NumPy · Matplotlib · KaggleHub · Google Colab

## Repository Contents

- `Chest-XRay.ipynb`: data loading, preprocessing, RNN and CNN models, training, evaluation, and prediction visualizations
- `docs/Deep_Learning_Course_Project_Slides.pdf`: final presentation slides

## Running It

1. Open `Chest-XRay.ipynb` in Google Colab or a local Jupyter environment.
2. Install dependencies if needed: `pip install torch torchvision kagglehub matplotlib numpy`
3. Run the notebook from top to bottom. The dataset is downloaded automatically through `kagglehub` (a Kaggle account may be required).

## Acknowledgments and Sources

- Kermany et al. (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, and the [Kaggle version](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- [A Deep Convolutional Neural Network for Pneumonia Detection in X-ray Images with Attention Ensemble](https://pmc.ncbi.nlm.nih.gov/articles/PMC10887593/) (An & Chen, 2024)
- [A Convolutional Neural Network ensemble model for Pneumonia Detection using chest X-ray images](https://www.sciencedirect.com/science/article/pii/S2772442523000436)
- [Detection of pneumonia using convolutional neural networks and deep learning](https://www.sciencedirect.com/science/article/pii/S0208521622000742)
- [Hybrid CNN-RNN for Medical Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728322/) (Munguia-Siu et al., 2024)
- [Enhancing Lung Disease Recognition through CNN-RNN Methodologies](https://doi.org/10.1038/s41598-026-45842-1) (Zahin et al., 2026)
- Lvov et al. (2024), *RNNs for Still Images*, Embedded World
- Background on pneumonia from the [CDC](https://www.cdc.gov/nchs/fastats/pneumonia.htm) and [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/pneumonia/diagnosis-treatment/drc-20354210)
