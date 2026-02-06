import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict, Any


def gauss_seidel_standard(
    f1: Callable[[float], float],
    f2: Callable[[float], float],
    f3: Callable[[float], float],
    f4: Callable[[float], float],
    h: float,
    epsilon: float,
) -> Tuple[np.ndarray, float, int]:
    n = int(1 / h)

    u = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]

    y = [j * h for j in range(n + 1)]
    x = [i * h for i in range(n + 1)]

    for j in range(n + 1):
        u[0][j] = f1(y[j])
        u[n][j] = f2(y[j])

    for i in range(n + 1):
        u[i][0] = f3(x[i])
        u[i][n] = f4(x[i])

    iteration = 0

    start_time = time.time()

    while True:
        max_diff = 0.0

        for i in range(1, n):
            for j in range(1, n):
                old_value = u[i][j]
                new_value = (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]) / 4
                u[i][j] = new_value
                max_diff = max(max_diff, abs(new_value - old_value))

        iteration += 1

        if max_diff < epsilon:
            break

    end_time = time.time()

    return np.array(u), end_time - start_time, iteration


def gauss_seidel_numpy(
    f1: Callable[[float], float],
    f2: Callable[[float], float],
    f3: Callable[[float], float],
    f4: Callable[[float], float],
    h: float,
    epsilon: float,
) -> Tuple[np.ndarray, float, int]:
    n = int(1 / h)
    u = np.zeros((n + 1, n + 1))

    y = np.arange(0, 1 + h, h)
    x = np.arange(0, 1 + h, h)

    u[0, :] = np.vectorize(f1)(y)
    u[n, :] = np.vectorize(f2)(y)
    u[:, 0] = np.vectorize(f3)(x)
    u[:, n] = np.vectorize(f4)(x)

    iteration = 0

    start_time = time.time()

    while True:
        u_old = u.copy()

        u[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]) / 4

        max_diff = np.max(np.abs(u - u_old))

        iteration += 1

        if max_diff < epsilon:
            break

    end_time = time.time()

    return u, end_time - start_time, iteration


def plot_solution(u: np.ndarray, h: float, title: str, filename: str):
    n = u.shape[0] - 1
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, u, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def run_experiment(
    h_values: List[float], epsilon_values: List[float], output_dir: str = "results"
) -> Dict[Tuple[float, float], Dict[str, Dict[str, Any]]]:
    def f1(y: float) -> float:
        return 1.0

    def f2(y: float) -> float:
        return y + 1.0

    def f3(x: float) -> float:
        return 1.0

    def f4(x: float) -> float:
        return x + 1.0

    import os

    os.makedirs(output_dir, exist_ok=True)

    results: Dict[Tuple[float, float], Dict[str, Any]] = {
        (h, eps): {
            "standard": {"time": 0.0, "iterations": 0},
            "numpy": {"time": 0.0, "iterations": 0},
            "convergence": 0.0,
        }
        for eps in epsilon_values
        for h in h_values
    }

    for h in h_values:
        for eps in epsilon_values:
            # Просто Python
            u_standard, time_standard, iterations_standard = gauss_seidel_standard(
                f1, f2, f3, f4, h, eps
            )

            # Реализация с NumPy
            u_numpy, time_numpy, iterations_numpy = gauss_seidel_numpy(
                f1, f2, f3, f4, h, eps
            )

            max_diff = np.max(np.abs(u_standard - u_numpy))

            results[(h, eps)]["standard"]["time"] = time_standard
            results[(h, eps)]["standard"]["iterations"] = iterations_standard

            results[(h, eps)]["numpy"]["time"] = time_numpy
            results[(h, eps)]["numpy"]["iterations"] = iterations_numpy

            results[(h, eps)]["convergence"] = max_diff

            title = f"h={h}, eps={eps}"
            plot_solution(u_standard, h, title, f"{output_dir}/h_{h}_eps_{eps}.png")
            print(f"h = {h}, eps = {eps} готово!")

    return results


def main():
    # Я 7 по списку, поэтому 7 вариант
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]

    results = run_experiment(h_values, epsilon_values, "results")

    print("\n                                                    === Результаты эксперимента ===")
    
    print("| h     | ε     | Время (Стандартная) | Итерации (Стандартная) | Время (NumPy) | Итерации (NumPy) | Максимальная разница | Ускорение |")
    print("| ----- | ----- | ------------------- | ---------------------- | ------------- | ---------------- | -------------------- | --------- |")

    for h in h_values:
        for eps in epsilon_values:
            time_standart = results[(h, eps)]["standard"]["time"]
            iterations_standart = results[(h, eps)]["standard"]["iterations"]

            time_numpy = results[(h, eps)]["numpy"]["time"]
            iterations_numpy = results[(h, eps)]["numpy"]["iterations"]

            convergence = results[(h, eps)]["convergence"]

            print(
                f"| {h:<5} | {eps:<5} | {time_standart:<19.2e} | {iterations_standart:<22} | {time_numpy:<13.4e} | {iterations_numpy:<16} | {convergence:<20.2e} | {(time_standart / time_numpy if time_numpy > 0.0 else 1.0):<9.3f} |"
            )


if __name__ == "__main__":
    main()
