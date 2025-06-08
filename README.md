# AI-for-software-Engineering-Week-2

![Alt text](./SKINCANCER.jpg)

# HAM10000 Dataset

## Human Against Machine with 10000 Training Images

### Overview

The HAM10000 dataset addresses a critical challenge in automated diagnosis of pigmented skin lesions: the small size and lack of diversity of available dermatoscopic image datasets. This comprehensive collection provides researchers and practitioners with a robust training set for academic machine learning purposes.

### Dataset Description

The HAM10000 dataset consists of **10,015 dermatoscopic images** collected from different populations and acquired using various modalities. This diversity ensures better generalization and representation across different imaging conditions and patient demographics.

### Diagnostic Categories

The dataset includes a representative collection of all important diagnostic categories in the realm of pigmented lesions:

- **Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)**
- **Basal cell carcinoma (bcc)**
- **Benign keratosis-like lesions (bkl)**
  - Solar lentigines
  - Seborrheic keratoses
  - Lichen-planus like keratoses
- **Dermatofibroma (df)**
- **Melanoma (mel)**
- **Melanocytic nevi (nv)**
- **Vascular lesions (vasc)**
  - Angiomas
  - Angiokeratomas
  - Pyogenic granulomas
  - Hemorrhage

### Ground Truth Validation

The dataset maintains high quality standards through multiple validation methods:

- **More than 50%** of lesions are confirmed through **histopathology (histo)** - the gold standard
- Remaining cases are validated through:
  - **Follow-up examination (follow_up)**
  - **Expert consensus (consensus)**
  - **Confirmation by in-vivo confocal microscopy (confocal)**

### Dataset Structure

- The dataset includes lesions with multiple images
- Lesions can be tracked using the **lesion_id column** within the `HAM10000_metadata` file
- This allows for proper handling of multiple views of the same lesion during training and evaluation

### Evaluation and Testing

- **The test set is not publicly available**
- An official evaluation server remains running for fair comparison of methods
- Researchers must submit their results to the official challenge website for evaluation
- **All publications using HAM10000 data should be evaluated on the official test set** to ensure fair and standardized comparison between methods

### Usage Guidelines

This dataset is intended for:
- Academic machine learning research
- Development of automated skin lesion diagnosis systems
- Training and validation of deep learning models for dermatoscopic image analysis

### Citation and Evaluation

When using this dataset in research:
1. Ensure your method is evaluated on the official test set via the challenge website
2. Follow proper academic citation guidelines
3. Compare results fairly with other methods using the same evaluation framework

### Files Included

- Dermatoscopic images (10,015 total)
- `HAM10000_metadata` file containing lesion tracking information and diagnostic labels
- Documentation of validation methods for each case

---

*For more information about the evaluation server and challenge details, please visit the official challenge website.*
