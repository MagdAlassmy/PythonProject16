import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def generate_dataset(m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.random.uniform(-1, 1, (m, 2))
    p1 = np.random.uniform(-1, 1, 2)
    p2 = np.random.uniform(-1, 1, 2)
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]
    scores = a * X[:, 0] + b * X[:, 1] + c
    y = np.sign(scores)
    y[y == 0] = 1
    w_f = np.array([a, b, c])
    return X, y, w_f

def perceptron_learning(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 10000
) -> Tuple[np.ndarray, int, List[int]]:
    m = len(X)
    X_bias = np.column_stack([np.ones(m), X])
    weights = np.zeros(3)
    steps = 0
    error_history: List[int] = []
    for _ in range(max_iter):
        scores = X_bias @ weights
        predictions = np.sign(scores)
        predictions[predictions == 0] = 1
        misclassified = np.where(predictions != y)[0]
        num_errors = len(misclassified)
        error_history.append(num_errors)
        if num_errors == 0:
            break
        idx = np.random.choice(misclassified)
        weights += alpha * (y[idx] - predictions[idx]) * X_bias[idx]
        steps += 1
    return weights, steps, error_history

def run_experiment(m: int, alpha: float, num_runs: int = 1000) -> float:
    all_steps = []
    for _ in range(num_runs):
        X, y, _ = generate_dataset(m)
        _, steps, _ = perceptron_learning(X, y, alpha=alpha)
        all_steps.append(steps)
    avg_steps = float(np.mean(all_steps))
    print(f"m={m}, alpha={alpha}: Durchschnittlich {avg_steps:.2f} Schritte")
    print(f"Std-Abweichung: {np.std(all_steps):.2f}")
    print(f"Min: {np.min(all_steps)}, Max: {np.max(all_steps)}")
    return avg_steps

def visualize_learning(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    weights, steps, error_history = perceptron_learning(X, y, alpha=alpha)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pos = X[y == 1]
    neg = X[y == -1]
    ax1.scatter(pos[:, 0], pos[:, 1], c='blue', marker='+', label='y=+1')
    ax1.scatter(neg[:, 0], neg[:, 1], c='red', marker='_', label='y=-1')
    x_line = np.linspace(-1, 1, 100)
    if abs(weights[2]) > 1e-10:
        y_line = -(weights[0] + weights[1] * x_line) / weights[2]
        ax1.plot(x_line, y_line, 'g-', label='Perzeptron-Grenze')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Daten & Entscheidungsgrenze")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(range(len(error_history)), error_history)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Anzahl Fehler")
    ax2.set_title("Lernkurve")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Benötigte Schritte: {steps}")
    return steps

if __name__ == "__main__":
    np.random.seed(42)
    avg_10_1 = run_experiment(m=10, alpha=1.0, num_runs=1000)
    avg_100_1 = run_experiment(m=100, alpha=1.0, num_runs=200)
    avg_100_01 = run_experiment(m=100, alpha=0.1, num_runs=200)
    avg_1000_1 = run_experiment(m=1000, alpha=1.0, num_runs=100)
    avg_1000_01 = run_experiment(m=1000, alpha=0.1, num_runs=100)
    X_viz, y_viz, _ = generate_dataset(20)
    visualize_learning(X_viz, y_viz, alpha=1.0)
