#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

namespace py = pybind11;

py::tuple gauss_seidel_cpp(
    py::function f1, py::function f2, py::function f3, py::function f4,
    double h, double epsilon
) {
    int n = static_cast<int>(1.0 / h);
    
    // Создаем массив NumPy
    py::array_t<double> u({n + 1, n + 1});
    auto u_unchecked = u.mutable_unchecked<2>();
    
    // Инициализация нулями
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            u_unchecked(i, j) = 0.0;
        }
    }

    // Подготовка координат
    std::vector<double> coord(n + 1);   
    for (int i = 0; i <= n; i++) coord[i] = i * h;
    
    // Граничные условия
    for (int j = 0; j <= n; j++) {
        u_unchecked(0, j) = f1(coord[j]).cast<double>();
        u_unchecked(n, j) = f2(coord[j]).cast<double>();
    }
    for (int i = 0; i <= n; i++) {
        u_unchecked(i, 0) = f3(coord[i]).cast<double>();
        u_unchecked(i, n) = f4(coord[i]).cast<double>();
    }
    
    int iteration = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (true) {
        double max_diff = 0.0;
        
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                // Запоминаем только ОДНО старое значение
                double old_val = u_unchecked(i, j);
                
                // Считаем новое значение
                double new_val = (u_unchecked(i - 1, j) + u_unchecked(i + 1, j) + 
                                  u_unchecked(i, j - 1) + u_unchecked(i, j + 1)) / 4.0;
                
                // Обновляем прямо в массиве
                u_unchecked(i, j) = new_val;
                
                // Считаем разницу для этой ячейки
                double diff = std::abs(new_val - old_val);
                if (diff > max_diff) max_diff = diff;
            }
        }
        
        iteration++;
        
        // Условие выхода
        if (max_diff < epsilon || iteration > 1000000) {
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    
    return py::make_tuple(u, elapsed_time, iteration);
}

PYBIND11_MODULE(gauss_seidel_cpp, m) {
    m.def("gauss_seidel_cpp", &gauss_seidel_cpp, "Gauss-Seidel solver optimized");
}