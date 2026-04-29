# DNSC 6330 — Individual Homework 05

## ML Security and Abuse Pathways: Adversarial Attacks on COMPAS

This repository contains the individual coding component for Lecture 05 of **DNSC 6330: Responsible Machine Learning**. The analysis uses the ProPublica COMPAS dataset to audit two predictive models, logistic regression and gradient-boosted trees, under adversarial machine learning conditions.

The work follows the Lecture 05 focus on ML security and abuse pathways, including deployment-time evasion, training-time poisoning, and privacy attacks through membership inference.

## Purpose

The purpose of this analysis is to test whether clean-model performance is enough to justify deployment in a high-stakes setting. The notebook starts from a COMPAS classification pipeline and evaluates how two models behave when exposed to adversarial pressure.

The analysis is organized into three main parts:

| Part | Focus | Key Methods |
|---|---|---|
| 1 | PGD evasion audit | Projected gradient descent attack, subgroup false-positive rates, AIR by race |
| 2 | Poisoning loop with fairness monitoring | Label-flip poisoning, AUC degradation, AIR degradation, PSI drift check |
| 3 | Membership inference depth | Shadow-model membership inference, ROC curves, confidence-gap histograms, regularization sweep |

The goal is not only to report model metrics, but also to interpret what those metrics mean for fairness, robustness, privacy, and responsible deployment.

## Python Libraries Used

The notebook uses the following Python libraries:

- `pandas` — data loading, cleaning, and tabular manipulation
- `numpy` — numerical operations and array handling
- `matplotlib` — plots for attack curves, ROC curves, confidence-gap histograms, and regularization sweeps
- `scikit-learn` — model pipelines, preprocessing, logistic regression, gradient-boosted trees, metrics, train/test split, and shadow models
- `scipy` — statistical and numerical support where needed

The notebook also defines custom helper functions for COMPAS preprocessing, false-positive-rate calculation by race, adverse impact ratio calculation, PGD-style tabular evasion attack, label-flip poisoning simulation, PSI-based drift monitoring, shadow-model membership inference, and the L2 regularization sweep for logistic regression.

## How to Reproduce

The repository should contain the following files:

- `Individual_Homework_05_Hesham Mushtaq_G23607459.ipynb` — analysis notebook
- `Individual_Homework_05_Written_Report_Hesham_Mushtaq_G23607459.pdf` — written report
- `README.md` — repository overview and reproduction guide

### Option 1: Google Colab

Open the notebook in Google Colab and run all cells from top to bottom. The notebook loads and processes the COMPAS data, trains the models, runs the attacks, and generates the result plots. Most required libraries are already available in the default Colab environment.

### Option 2: Local Jupyter Notebook

Clone the repository, install the required libraries, and open the notebook locally:

`pip install pandas numpy scikit-learn scipy matplotlib jupyter`

Then launch Jupyter:

`jupyter notebook`

Open `Individual_Homework_05_Hesham Mushtaq_G23607459.ipynb` and run all cells from the top. The notebook is designed to be reproducible with fixed random seeds used in the train/test split and model-training steps. Minor numerical differences may occur depending on package versions.

## Interpretation of Results

### Clean Baseline

Before any adversarial attack, the models already show unequal error rates by race. The logistic regression model has a test AUC of about **0.735**, with false-positive rates of **0.281** for African-American defendants and **0.143** for Caucasian defendants. The adverse impact ratio is **1.961**.

The gradient-boosted tree model has a test AUC of about **0.718**, with false-positive rates of **0.317** for African-American defendants and **0.178** for Caucasian defendants. Its adverse impact ratio is **1.782**.

This means the adversarial analysis begins from an already imbalanced baseline. The attacks do not create the fairness problem from nothing; they stress a model that is already uneven across racial groups.

### Part 1 — PGD Evasion Audit

Both models are vulnerable to PGD evasion attacks. As the perturbation budget increases, subgroup false-positive rates rise sharply.

For logistic regression, false-positive rates increase from **0.569 / 0.370** for African-American and Caucasian defendants at `epsilon = 0.25`, to **0.791 / 0.560** at `epsilon = 0.50`, to **0.978 / 0.884** at `epsilon = 1.00`. By `epsilon = 2.00`, both groups reach **FPR = 1.000**, meaning complete failure.

For the gradient-boosted tree, false-positive rates increase from **0.553 / 0.351** at `epsilon = 0.25`, to **0.733 / 0.514** at `epsilon = 0.50`, to **0.899 / 0.760** at `epsilon = 1.00`. By `epsilon = 2.00`, both groups also reach **FPR = 1.000**.

The gradient-boosted tree is somewhat less vulnerable than logistic regression at intermediate attack strengths, but neither model is robust enough for high-stakes deployment without additional protections. AIR moves closer to 1.0 under severe attack, but this is not fairness improvement. It is parity of failure, where both racial groups are being pushed toward the same harmful outcome. This is why subgroup false-positive rates must be monitored alongside AIR.

### Part 2 — Poisoning Loop with Fairness Monitoring

The label-flip poisoning experiment shows that AUC alone is not a reliable monitoring metric. Across both target-race variants, test AUC stays nearly flat, roughly between **0.730 and 0.735**. A reviewer watching only AUC would likely conclude that the model is stable.

AIR tells a different story. When African-American defendants are targeted, AIR rises from **1.961** at baseline to **3.010** at a 30% poison rate. The Caucasian-targeted variant stays closer to the baseline range, around **1.84 to 2.04**.

The stealth-zone finding is important. The clean baseline already violates the fairness band of `[0.80, 1.25]`, and AUC remains stable during poisoning. Therefore, the full tested range acts like a stealth regime: aggregate performance appears stable while fairness harm remains outside the acceptable band.

A PSI-based drift monitor also fails because label-flip poisoning changes labels, not feature distributions. Feature-level PSI values remain at **max PSI = 0.000**. This means AUC monitoring and PSI drift monitoring together would still miss the poisoning attack.

### Part 3 — Membership Inference Depth

The membership inference attack does not produce meaningful separation between training members and non-members. The MI AUC is approximately **0.497** for logistic regression and **0.500** for the gradient-boosted tree. These values are essentially random guessing.

The ROC curves sit close to the diagonal, and the confidence-gap histograms overlap heavily. The result is directionally consistent with the lecture concept that overfitting can increase membership-inference risk. The gradient-boosted tree has the larger generalization gap and the slightly higher MI AUC, while logistic regression has the smaller gap and slightly lower MI AUC. However, with only two models, this is a directional sanity check rather than a statistical test.

The L2 regularization sweep for logistic regression over `C = {0.01, 0.1, 1.0, 10.0}` keeps MI AUC close to 0.50. Stronger regularization does not produce a meaningful privacy gain in this setup, so sacrificing predictive utility for regularization would not be justified by these results alone.

## Overall Recommendation

The highest-risk finding is the PGD evasion result. A deployment-time attacker can sharply increase false positives without retraining the model or changing the training data.

A proactive mitigation is to include adversarial stress testing during model selection. In this analysis, choosing the gradient-boosted tree over logistic regression reduces attacked false-positive rates at `epsilon = 1.0` by about **7.9 percentage points for African-American defendants** and **12.4 percentage points for Caucasian defendants**. This helps, but it does not solve the robustness problem.

A reactive mitigation is to monitor subgroup false-positive rates and AIR alongside aggregate AUC. The poisoning results show why this is necessary: AUC stays almost flat while AIR moves to harmful levels and PSI remains at zero.

The final responsible-ML takeaway is that robustness and fairness controls must be used together. Model substitution alone does not remove the disparate-impact pattern, and fairness monitoring detects harm after it appears rather than preventing it.


## Files

- `Individual_Homework_05_Hesham Mushtaq_G23607459.ipynb` — analysis notebook with code, attack simulations, plots, and metrics
- `Individual_Homework_05_Written_Report_Hesham_Mushtaq_G23607459.pdf` — concise written report interpreting the results
- `README.md` — repository overview, reproduction instructions, and summary of findings
