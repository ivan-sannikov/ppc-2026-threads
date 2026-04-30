#include "../include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <complex>
#include <numeric>
#include <thread>
#include <vector>

#include "../../common/include/common.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_mult_matrix_complex_col {

struct SparseColumn {
  std::vector<int> rows;
  std::vector<std::complex<double>> vals;
};

ShvetsovaKMultMatrixComplexSTL::ShvetsovaKMultMatrixComplexSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = MatrixCCS(0, 0, std::vector<int>{0}, std::vector<int>{}, std::vector<std::complex<double>>{});
}

bool ShvetsovaKMultMatrixComplexSTL::ValidationImpl() {
  return true;
}

bool ShvetsovaKMultMatrixComplexSTL::PreProcessingImpl() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  auto &matrix_c = GetOutput();
  matrix_c.rows = matrix_a.rows;
  matrix_c.cols = matrix_b.cols;
  matrix_c.row_ind.clear();
  matrix_c.values.clear();
  matrix_c.col_ptr.clear();
  matrix_c.col_ptr.push_back(0);
  return true;
}

bool ShvetsovaKMultMatrixComplexSTL::RunImpl() {
  const MatrixCCS &matrix_a = std::get<0>(GetInput());
  const MatrixCCS &matrix_b = std::get<1>(GetInput());

  auto &matrix_c = GetOutput();

  std::vector<SparseColumn> columns_c(matrix_b.cols);

  auto compute_column = [&](int i, std::vector<std::complex<double>> &column_c) {
    std::ranges::fill(column_c, std::complex<double>{0.0, 0.0});

    for (int j = matrix_b.col_ptr[i]; j < matrix_b.col_ptr[i + 1]; j++) {
      int tmp_ind = matrix_b.row_ind[j];
      std::complex<double> tmp_val = matrix_b.values[j];
      for (int ind = matrix_a.col_ptr[tmp_ind]; ind < matrix_a.col_ptr[tmp_ind + 1]; ind++) {
        int row = matrix_a.row_ind[ind];
        std::complex<double> val_a = matrix_a.values[ind];
        column_c[row] += tmp_val * val_a;
      }
    }
    for (int index = 0; std::cmp_less(index, column_c.size()); ++index) {
      if (column_c[index].real() != 0.0 || column_c[index].imag() != 0.0) {
        columns_c[i].rows.push_back(index);
        columns_c[i].vals.push_back(column_c[index]);
      }
    }
  };

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 4;  // Запасной вариант, если система не вернула количество потоков
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  int cols_per_thread = matrix_b.cols / num_threads;
  unsigned int remainder = matrix_b.cols % num_threads;

  int current_start = 0;
  for (unsigned int t = 0; t < num_threads; ++t) {
    int start_col = current_start;
    int end_col = start_col + cols_per_thread + (t < remainder ? 1 : 0);
    current_start = end_col;

    if (start_col < end_col) {
      threads.emplace_back([&, start_col, end_col]() {
        std::vector<std::complex<double>> column_c_local(matrix_a.rows, {0.0, 0.0});
        for (int i = start_col; i < end_col; ++i) {
          compute_column(i, column_c_local);
        }
      });
    }
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  for (int i = 0; i < matrix_b.cols; i++) {
    matrix_c.row_ind.insert(matrix_c.row_ind.end(), columns_c[i].rows.begin(), columns_c[i].rows.end());
    matrix_c.values.insert(matrix_c.values.end(), columns_c[i].vals.begin(), columns_c[i].vals.end());
    matrix_c.col_ptr.push_back(static_cast<int>(matrix_c.row_ind.size()));
  }

  return true;
}

bool ShvetsovaKMultMatrixComplexSTL::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_mult_matrix_complex_col
