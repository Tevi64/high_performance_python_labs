import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict, Any
import os
from numba import njit  # type: ignore

from gauss_seidel_cpp import gauss_seidel_cpp


@njit(fastmath=True)
def _numba_kernel(u: np.ndarray, n: int, epsilon: float) -> int:
    iteration = 0
    while True:
        max_diff = 0.0
        for i in range(1, n):
            for j in range(1, n):
                old_value = u[i, j]
                new_value = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]) / 4.0
                u[i, j] = new_value

                diff = abs(new_value - old_value)
                if diff > max_diff:
                    max_diff = diff

        iteration += 1
        if max_diff < epsilon or iteration > 1000000:
            break
    return iteration


def gauss_seidel_numba(
    f1: Callable[[float], float],
    f2: Callable[[float], float],
    f3: Callable[[float], float],
    f4: Callable[[float], float],
    h: float,
    epsilon: float,
) -> Tuple[np.ndarray, float, int]:
    n = int(1 / h)
    u = np.zeros((n + 1, n + 1))

    # Координаты
    y = np.linspace(0, 1, n + 1)
    x = np.linspace(0, 1, n + 1)

    # Граничные условия
    for j in range(n + 1):
        u[0, j] = f1(y[j])
        u[n, j] = f2(y[j])
    for i in range(n + 1):
        u[i, 0] = f3(x[i])
        u[i, n] = f4(x[i])

    # Разогрев Numba (компиляция происходит при первом вызове)
    # Вызываем один раз с маленьким лимитом, чтобы не считать долго во время замера
    _numba_kernel(u.copy(), n, 1.0)

    start_time = time.time()
    # Основной расчет
    iterations = _numba_kernel(u, n, epsilon)
    end_time = time.time()

    return u, end_time - start_time, iterations


def plot_solution(u: np.ndarray, h: float, title: str, filename: str) -> None:
    n = u.shape[0] - 1
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, u.T, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def run_experiment(
        h_values: List[float],
        epsilon_values: List[float],
        output_dir: str = "results"
) -> Dict[Tuple[float, float], Any]:
    def f1(y: float) -> float:
        return 1.0

    def f2(y: float) -> float:
        return y + 1.0

    def f3(x: float) -> float:
        return 1.0

    def f4(x: float) -> float:
        return x + 1.0

    os.makedirs(output_dir, exist_ok=True)

    results: Dict[Tuple[float, float], Any] = {}

    for h in h_values:
        for eps in epsilon_values:
            # Numba
            u_numba, time_numba, iterations_numba = gauss_seidel_numba(
                f1, f2, f3, f4, h, eps
            )

            # C++ и PyBind11
            u_cpp, time_cpp, iterations_cpp = gauss_seidel_cpp(
                f1, f2, f3, f4, h, eps
            )

            results[(h, eps)] = {
                "cpp": (time_cpp, iterations_cpp),
                "numba": (time_numba, iterations_numba),
                "convergence": np.max(np.abs(u_cpp - u_numba))

            }

            title = f"h={h}, eps={eps}"
            plot_solution(u_numba, h, title, f"{output_dir}/h_{h}_eps_{eps}.png")
            print(f"h = {h}, eps = {eps} готово!")

    return results


def main() -> None:
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]

    results = run_experiment(h_values, epsilon_values)

    print(
        "\n"
        "                                                    === Результаты эксперимента ===\n"
        "\n"
        "| h     | ε     | Время (Numba) | Итерации (Numba) | Время (C++) | Итерации (C++) | Максимальная разница | Ускорение |\n"
        "| ----- | ----- | ------------- | ---------------- | ----------- | -------------- | -------------------- | --------- |"
    )

    for (h, eps), res in results.items():
        time_numba, iterations_numba = results[(h, eps)]["numba"]
        time_cpp, iterations_cpp = results[(h, eps)]["cpp"]
        convergence = results[(h, eps)]["convergence"]
        speedup = time_numba / time_cpp if time_cpp > 0 else 0

        print(
            f"| {h:<5} | {eps:<5} | {time_cpp:<13.2e} | {iterations_cpp:<16} | {time_cpp:<11.2e} | {iterations_cpp:<14} | {convergence:<20.2e} | {speedup:<9.1f} |"
        )


if __name__ == "__main__":
    main()
