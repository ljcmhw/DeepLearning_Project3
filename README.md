# Deep Learning Project 3 · Adversarial Attacks and Transferability  
*Spring 2025 · Principles of Deep Learning*

---

## 🧠 1. Project Overview

This project explores adversarial robustness and model transferability by performing multiple white-box attacks on a pretrained image classification model (e.g., ResNet-34). We implement the following:

- ✅ **Task 1**: Evaluate baseline performance on clean images  
- ⚡ **Task 2**: FGSM (Fast Gradient Sign Method) attack  
- 🔁 **Task 3**: Iterative / PGD (Projected Gradient Descent) attack  
- 🧱 **Task 4**: Patch-based adversarial attack  
- 🔀 **Task 5**: Cross-model transferability evaluation

The project is implemented using PyTorch and torchvision, with fully automated scripts to reproduce all experiments.

---

## 📁 2. Repository Structure

.
├── data/ # ⇦ test sets (clean & adversarial)
│ ├── TestDataSet/ # original clean test set (ImageFolder format)
│ ├── adv1/ # FGSM adversarial set
│ ├── adv2/ # PGD adversarial set
│ ├── adv3/ # Patch adversarial set
│ └── labels.json # (optional) class label names
├── scripts/ # all runnable experiment scripts
│ ├── evaluate.py
│ ├── fgsm_attack.py
│ ├── improved_attack.py
│ ├── patch_attack.py
│ └── transfer_attack.py
├── notebooks/ # (optional) Jupyter notebooks for analysis
├── requirements.txt # Python dependencies
└── README.md # you are here

---

## ⚙️ 3. Environment Setup

```bash
git clone https://github.com/<your-username>/project3-jailbreak.git
cd project3-jailbreak

# (Recommended) Create virtual environment
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
