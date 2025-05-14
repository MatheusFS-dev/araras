# Description

This document is a concise compilation of tips, theoretical explanations, and practical guidance aiding developers in setting up and managing their development environments, optimizing workflows, and understanding key concepts in machine learning and software engineering.


# Table of Contents

- [Description](#description)
- [Table of Contents](#table-of-contents)
  - [Regularizers](#regularizers)
  - [Multi-objective Optimization](#multi-objective-optimization)
    - [How Optuna handles multi-objective studies](#how-optuna-handles-multi-objective-studies)
  - [Multi-class classification vs multi-label classification](#multi-class-classification-vs-multi-label-classification)
  - [Fine-tuning a trained model with SGD](#fine-tuning-a-trained-model-with-sgd)
  - [About Batch Normalization in DNNs](#about-batch-normalization-in-dnns)
    - [When BN may be overkill or even harmful](#when-bn-may-be-overkill-or-even-harmful)

---

## Regularizers

| Regularizer             | What It Does                                                                                              | When to Use                                                                                   | Benefits                                                                                                  | Disadvantages                                                              |
|-------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **L1**                  | Adds the sum of the absolute values of weights (‖w‖₁) to the loss, encouraging many weights to become zero.  | Use when sparsity is desired, especially in high-dimensional settings with many irrelevant features.  | Promotes sparsity, acts as feature selection, and can lead to more interpretable models.                 | Non-smooth gradients at zero; may cause optimization instability.          |
| **L2**                  | Adds the sum of squared weights (‖w‖₂²) to the loss, discouraging large weights via quadratic penalization.  | Common default in neural networks to control model complexity and ensure smooth optimization.      | Provides smooth gradients, improves generalization by keeping weights small, and is computationally efficient. | Does not yield sparse solutions; models remain dense.                      |
| **L1L2 (Elastic Net)**  | Combines L1 and L2 penalties to balance sparsity and weight decay.                                         | Use when you need both sparsity and stability, particularly with correlated features.             | Balances feature selection and smooth optimization; hyperparameters allow flexible tuning.               | Increases complexity in hyperparameter tuning; requires balancing two penalties. |
| **Orthogonal**          | Adds a penalty that encourages weight matrices to be orthogonal (penalizing the deviation of \\(W^T W\\) from the identity). | Ideal for deep or recurrent networks where diverse features and stable gradient flow are crucial. | Promotes diversity among neurons, reduces redundancy, and improves gradient flow in deep architectures.    | Computationally more expensive and adds extra hyperparameter tuning requirements.  |

Regularization becomes critical when your model’s capacity (number of parameters) starts to greatly exceed the information content of your data (number of training examples). For example, considering a DNN structure:

| Network / Data Regime                                   | Recommendation                                                                                                                                                                                             |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tiny net**<br/>(< 5 layers, < 100 K parameters)       | • **Skip heavy weight decay**: L2 coefficients (λ) ≈ 0 or very small (1e-6)…1e-5<br/>• **Light dropout**: 0–0.1 only if you see overfitting<br/>• Data ≥ 10× parameters → minimal reg needed               |
| **Small to medium**<br/>(5–10 layers, 100 K–1 M params) | • **L2 weight decay**: λ≈1e-4…1e-3 to combat moderate overfitting<br/>• **Dropout**: 0.1–0.3 on hidden layers<br/>• **Optional L1** (λ≈1e-5) if you need feature sparsity                     |
| **Aggressive regime**<br/>(8–10 layers, > 1 M params)   | • **L2 weight decay**: λ≈1e-5…1e-4 when paired with AdamW-style decay<br/>• **Higher dropout**: 0.3–0.5<br/>• Consider **L1/L2** mix (e.g. L1L2) for both sparsity and smooth weights |
| **Small dataset**<br/>(< 10 K examples)                 | • **Always use regularization**: parameter-to-sample ratio > 1 → strong reg (L2 λ≈1e-3…1e-2)<br/>• **Dropout**: 0.2–0.5<br/>• Early stopping becomes crucial                                               |
| **Large dataset**<br/>(> 100 K examples)                | • You can **dial back** weight decay (λ≈1e-6…1e-4) and dropout (0–0.2)<br/>• Model can “eat” parameters if data supports it—overfitting risk lower                                                         |


* **Capacity vs. data**: If #parameters ≫ #examples, the model can memorize; regularizers (L2/L1/dropout) inject bias/noise to force learning meaningful patterns.
* **Dropout** is a stochastic regularizer: stronger dropout (0.3–0.5) for larger nets or smaller datasets; lighter dropout (0–0.2) when you suspect under-regularization only in final layers.
* **Weight decay (L2)** smooths weight values and penalizes large weights; safer default for most medium-sized DNNs.
* **L1** encourages sparsity—useful if you believe only a subset of features matters or to compress very wide layers.

---

## Multi-objective Optimization

Multi-objective optimization tackles problems where you must optimize two or more conflicting criteria at once—say, minimizing validation loss *and* model size. Unlike single-objective search, you don’t get one “best” trial but rather a **Pareto front**: the set of non-dominated solutions where improving one objective always degrades another.

### How Optuna handles multi-objective studies

1. **Define multiple directions.**  
   When you create the study, pass a list of directions—one per objective:
   ```python
   study = optuna.create_study(
       directions=["minimize", "minimize"]  # e.g., (loss, size) both to minimize
   )
   ```  

2. **Return a sequence of values.**  
   Your `objective` must return a tuple (or list) of floats, matching the number and order of `directions`. For example:  
   ```python
   def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> Sequence[float]:
       # Hyperparameters…
       model = build_model(trial)
       loss = train_and_evaluate(model)
       param_count = model.count_params()
       return loss, float(param_count)
   ```  
   Optuna automatically treats the first return as objective 1 and the second as objective 2.

---

## Multi-class classification vs multi-label classification

- **Multi-class classification**: Each instance belongs to one and only one class. For example, classifying images of animals into categories like "cat," "dog," or "bird." The model outputs a single label for each instance.
- **Multi-label classification**: Each instance can belong to multiple classes simultaneously. For example, tagging an image with multiple labels like "cat," "cute," and "pet." The model outputs a set of labels for each instance.
- **Key Differences**:
  - **Output Layer**: Multi-class uses softmax for a single label, while multi-label uses sigmoid for independent probabilities.
  - **Loss Function**: Multi-class typically uses categorical cross-entropy, while multi-label uses binary cross-entropy.
  - **Evaluation Metrics**: Multi-class often uses accuracy or F1-score, while multi-label may use hamming loss or subset accuracy.
- **Use Cases**: Multi-class is for exclusive categories, while multi-label is for overlapping categories.

---

## Fine-tuning a trained model with SGD

Usually SGD does not work well with the first version of the model, but it could help to fine-tune a trained model. The SGD is very noisy and can help to escape local minima.


---

## About Batch Normalization in DNNs

Batch normalization (BN) remains a valuable tool for deep feed-forward nets—rarely “redundant” in practice—but its benefit depends on depth, batch size, and data preprocessing:

* **Accelerated convergence & smoother optimization**
  BN standardizes each layer’s inputs (zero mean, unit variance) per mini-batch, which was originally motivated as reducing “internal covariate shift” but is now understood to **smooth the loss landscape**, yielding more stable, predictable gradients and allowing higher learning rates.

* **Implicit regularization**
  The noise in batch statistics injects stochasticity, which often **improves generalization** and can reduce reliance on dropout.

* **Stability in deeper nets**
  In networks beyond ∼5–10 layers, BN **mitigates vanishing/exploding gradients** and sensitivity to initialization, making very deep stacks trainable.

### When BN may be overkill or even harmful

| Scenario                              | Why BN adds little or hurts                                        | Alternative                                       |
| ------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------- |
| **Shallow nets (≤ 5 layers)**         | Few internal transforms → small covariate-shift issues             | Simple input standardization (zero-mean/unit-var) |
| **Small/variable batch sizes (< 16)** | Noisy mean/var estimates degrade performance ([Stack Overflow][3]) | LayerNorm or GroupNorm (batch-independent)        |
| **Ultra-low-latency inference**       | Maintains running-mean/var overhead; two‐mode logic                | Remove BN; rely on fixed preprocessing            |


* **Use BN** if you train with batches ≥ 32 and your network is deeper than ∼5 layers—it will speed convergence and add mild regularization.
* **Skip BN** (or switch to LayerNorm) if your batches are very small/irregular, or if you only have a shallow (≤ 5) stack on already normalized inputs.
