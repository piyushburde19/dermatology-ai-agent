# AI Agents for Dermatology Image Analysis

AI-assisted dermatology image analysis using **EfficientNet-B0**, uncertainty-aware routing, **Grad-CAM explainability**, and a coordinated multi-agent workflow.

> **Important:** This is an academic/research prototype. It is not a definitive medical diagnosis and must not replace evaluation by a qualified healthcare professional.

## Overview

The project combines deep-learning image classification with multiple specialized agents:

```text
User
  ↓
React Web Interface
  ↓
FastAPI Backend
  ↓
Dermatology Agent Orchestrator
  ├── Image Analysis Agent
  │     ├── EfficientNet-B0 classification
  │     ├── Class probabilities
  │     ├── OOD/support check
  │     └── Grad-CAM
  │
  ├── Risk & Uncertainty Agent
  │     ├── Confidence
  │     ├── Feature similarity
  │     ├── Entropy
  │     └── Review routing
  │
  ├── Explanation Agent
  │     └── Human-readable interpretation
  │
  └── Report Agent
        └── Combined analysis report
```

The goal is not simply to classify an image. The system also decides when a prediction should be treated cautiously, provides an explanation, generates visual evidence with Grad-CAM, and combines the outputs into a structured report.

## Main Features

- EfficientNet-B0 based six-class dermatology image classification
- Patient-level train/validation/test splitting
- Confidence and uncertainty assessment
- Out-of-distribution (OOD) / unsupported-image routing
- Human-review recommendation
- Grad-CAM visual explainability
- Multi-agent orchestration
- Rule-based AI Assistant for discussing the current analysis
- FastAPI REST backend
- React + Vite frontend
- Automated validation scripts for classes, OOD routing, and review routing

## Supported Classes

The model was trained for six PAD-UFES-20 diagnostic classes:

| Code | Class |
|---|---|
| ACK | Actinic Keratosis |
| BCC | Basal Cell Carcinoma |
| MEL | Melanoma |
| NEV | Nevus |
| SCC | Squamous Cell Carcinoma |
| SEK | Seborrheic Keratosis |

## Dataset

The project uses the **PAD-UFES-20** dataset.

Dataset summary used in this project:

- 2,298 images
- 1,373 patients
- 1,641 lesions
- 6 diagnostic classes
- Smartphone/clinical dermatology images
- Patient-level splitting to avoid patient overlap between train, validation, and test sets

### Dataset location

The dataset is intentionally **not stored in this Git repository**.

After obtaining the dataset, the expected local structure is:

```text
data/
└── raw/
    └── PAD-UFES-20/
        ├── metadata.csv
        ├── metadata_split.csv
        └── images/
            ├── imgs_part_1/
            ├── imgs_part_2/
            └── imgs_part_3/
```

Do not commit the dataset to GitHub.

## Dataset Split

Patient-level split used by the project:

| Split | Images | Patients |
|---|---:|---:|
| Train | 1470 | 880 |
| Validation | 368 | 220 |
| Test | 460 | 273 |

Patient overlap between the three splits was checked and was zero.

## Model

### EfficientNet-B0

The image analysis model uses pretrained **EfficientNet-B0** with a six-class classification head.

Input:

```text
224 × 224 RGB
```

ImageNet normalization is used.

Training included:

- Horizontal/vertical augmentation
- Rotation
- Color jitter
- AdamW optimization
- Automatic mixed precision
- Early stopping
- Class-weighted training
- Fine-tuning of the EfficientNet backbone

## Experimental Results

The following results were obtained during the project's sequential experiments:

| Experiment | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| Frozen EfficientNet-B0 | 42.83% | 0.3628 |
| Fine-tuning | 60.43% | 0.4480 |
| Inverse-frequency weighting | 62.83% | 0.5058 |
| Moderated class weighting | **65.00%** | **0.5690** |

The final moderated experiment achieved:

```text
Test Accuracy : 65.00%
Test Macro F1 : 0.5690
```

The experiments were performed sequentially, so later experiments were continued from earlier checkpoints rather than being completely independent random restarts.

## OOD and Uncertainty Routing

The system uses three research-oriented signals:

- Feature similarity
- Prediction confidence
- Prediction entropy

Current research heuristic:

```text
Similarity threshold : 0.37
Confidence threshold : 0.85
Entropy threshold    : 0.60
```

An image may be routed as:

```text
Supported
Review Recommended
Unsupported
```

### Test-set OOD validation

On the 460-image test set:

```text
Test images             : 460
Supported images        : 403
OOD / rejected images   : 57
OOD rejection rate      : 12.39%
Accuracy on supported   : 68.24%
```

These thresholds are **research heuristics and are not clinically validated**.

## Human-Review Routing

For supported test images, the system can recommend human review when uncertainty signals are present.

The routing considers:

- Low confidence
- High entropy
- Low feature similarity

Multiple signals can occur for the same image.

The review-routing validation identified:

```text
Supported images              : 403
Review Recommended            : 119
Supported without review      : 284
```

## Grad-CAM Explainability

For supported predictions, the Image Analysis Agent can generate a Grad-CAM visualization from the final EfficientNet feature layer.

The output helps visualize image regions that contributed to the model's prediction.

Grad-CAM is used as an **interpretability aid**. It does not prove that a prediction is medically correct.

## Multi-Agent Architecture

### 1. Image Analysis Agent

Responsible for:

- Loading the uploaded image
- Running the EfficientNet model
- Producing probabilities
- Producing confidence
- Calculating OOD-related metrics
- Generating Grad-CAM for supported images

### 2. Risk & Uncertainty Agent

Responsible for:

- Evaluating confidence
- Evaluating feature similarity
- Evaluating entropy
- Detecting OOD status
- Deciding whether human review is recommended

### 3. Explanation Agent

Responsible for:

- Converting model output into human-readable language
- Explaining the predicted class
- Providing interpretation
- Providing a next-step recommendation

### 4. Report Agent

Responsible for:

- Combining outputs from the other agents
- Producing the final structured report
- Adding the research/medical disclaimer

### Orchestrator

The **Dermatology Agent Orchestrator** coordinates the four agents in sequence:

```text
Image Analysis
      ↓
Risk & Uncertainty
      ↓
Explanation
      ↓
Report
```

## Project Structure

```text
dermatology-ai-agent/
│
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── api/
│   │   ├── core/
│   │   ├── ml/
│   │   ├── models/
│   │   ├── services/
│   │   ├── utils/
│   │   └── main.py
│   ├── .env.example
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   ├── uploads/
│   └── ood/
│
├── models/
│   ├── checkpoints/
│   └── gradcam/
│
├── tests/
├── docs/
├── notebooks/
├── .gitignore
├── docker-compose.yml
└── README.md
```

## Requirements

Recommended environment:

```text
Windows 10/11
Python 3.11+
Node.js 20+
Git
```

A CUDA-capable NVIDIA GPU is recommended for faster inference/training, but the application can also run on CPU with appropriate PyTorch installation.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/piyushburde19/dermatology-ai-agent.git
cd dermatology-ai-agent
```

### 2. Create the Python virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### 3. Install PyTorch

Install a PyTorch build appropriate for the friend's hardware from the official PyTorch installation selector.

Then install the backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

### 4. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## Model Checkpoint

The trained model checkpoint is intentionally **not stored in GitHub** because model binaries are large.

Expected file:

```text
models/checkpoints/efficientnet_b0_moderated_best.pth
```

The checkpoint must be copied into that exact location before using the trained prediction pipeline.

If the checkpoint is unavailable, the application cannot perform the trained dermatology prediction.

## Environment File

The backend uses environment variables where required.

Copy:

```text
backend/.env.example
```

to:

```text
backend/.env
```

Do not commit `.env`.

## Running the Backend

From the project root, with `.venv` activated:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

## Running the Frontend

Open a second terminal:

```powershell
cd frontend
npm run dev
```

The Vite development server will display the local frontend URL, normally:

```text
http://localhost:5173
```

## Basic Demo Flow

1. Start the FastAPI backend.
2. Start the React frontend.
3. Open the frontend.
4. Upload a dermatology image.
5. The Image Analysis Agent runs the model.
6. The Risk & Uncertainty Agent evaluates the result.
7. The Explanation Agent generates an interpretation.
8. The Report Agent creates the final report.
9. Grad-CAM is displayed for supported images.
10. The AI Assistant can answer questions about the current analysis.

## Testing

Run the multi-class API validation:

```powershell
python -m tests.test_multi_class_api
```

Run review-routing validation:

```powershell
python -m tests.test_review_routing
```

Run full test-set OOD validation:

```powershell
python -m tests.validate_test_ood
```

Run threshold analysis:

```powershell
python -m tests.threshold_analysis
```

## Important Limitations

- The dataset is relatively small for a six-class medical-image classification task.
- Class imbalance remains a challenge.
- Some legitimate test images can be routed as unsupported by the current OOD heuristic.
- SCC performance remains weaker than several other classes.
- Model confidence should not be interpreted as disease probability.
- Grad-CAM provides visual interpretability but does not prove prediction correctness.
- OOD thresholds are research heuristics and are not clinically validated.
- The system has not been clinically validated or approved for medical diagnosis.
- Performance on external images may differ from performance on PAD-UFES-20.

## Future Scope

Potential future improvements include:

- Larger and more diverse dermatology datasets
- External clinical validation
- Better OOD detection
- Calibration of prediction probabilities
- Improved minority-class performance
- More advanced segmentation
- Additional clinical metadata integration
- Retrieval-augmented medical knowledge
- More sophisticated agent planning and tool use
- Prospective clinical evaluation

## Academic Project

**Project:** AI Agents for Dermatology Image Analysis

**Branch:** Artificial Intelligence

**Application:** AI-assisted dermatology image analysis

## Disclaimer

This software is an academic/research prototype. It is intended for educational and research purposes only. It does not provide a definitive medical diagnosis, and users should consult a qualified healthcare professional for medical evaluation.
