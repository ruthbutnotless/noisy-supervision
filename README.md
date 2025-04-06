# noisy-supervision
A simple PyTorch experiment demonstrating the impact of noisy supervision on model learning.

# Noisy Supervision in Model Learning

This project demonstrates the **impact of noisy supervision** in a simple **PyTorch experiment**. Noisy supervision is a common challenge in real-world machine learning applications where labels may be inaccurate or incomplete. This experiment, inspired by the paper *Impact of Noisy Supervision in Foundation Model Learning*, explores how label noise affects model training and performance.

## 📌 Features

-   Uses **CIFAR-10** dataset to compare model training on **clean vs. noisy labels**. Noisy labels are generated by randomly flipping a percentage of the original labels.
-   Implements a **basic feedforward neural network** using **PyTorch**.
-   Provides **visualizations** of accuracy trends over epochs.

## 🚀 Setup Instructions

1.  Clone this repository:

    ```bash
    git clone https://github.com/ruthbutnotless/noisy-supervision.git
    cd noisy-supervision
    ```

2.  Install dependencies:

    ```bash
    pip install torch torchvision matplotlib numpy
    ```

3.  Run the experiment:

    ```bash
    python noisy_supervision.py
    ```

    **Results**:
    -   The model is trained with both clean and noisy labels, and their performances are compared. Key hyperparameters include a learning rate of 0.001, a batch size of 64, and 50 training epochs.
    -   Accuracy plots are generated to visualize the effect of noise. These plots typically show a slower convergence or a lower final accuracy for models trained with noisy labels.

    Here is an example of the generated accuracy plots:

    ![Accuracy Plot](accuracy_plot.png)

## Reference

-   Hao Chen, Zihan Wang, et al. Impact of Noisy Supervision in Foundation Model Learning (DOI: 10.1109/TPAMI.2025.3552309).
-   PyTorch Documentation