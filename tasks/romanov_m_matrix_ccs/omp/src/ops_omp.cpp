#include "romanov_m_matrix_ccs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "util/include/util.hpp"

namespace romanov_m_matrix_ccs {

RomanovMMatrixCCSOMP::RomanovMMatrixCCSOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RomanovMMatrixCCSOMP::ValidationImpl() {
  return GetInput().first.cols_num == GetInput().second.rows_num;
}

bool RomanovMMatrixCCSOMP::PreProcessingImpl() {
  return true;
}

bool RomanovMMatrixCCSOMP::RunImpl() {
  const auto &A = GetInput().first;
  const auto &B = GetInput().second;
  auto &C = GetOutput();

  C.rows_num = A.rows_num;
  C.cols_num = B.cols_num;
  C.col_ptrs.assign(C.cols_num + 1, 0);

  std::vector<std::vector<double>> temp_vals(C.cols_num);
  std::vector<std::vector<size_t>> temp_rows(C.cols_num);

#pragma omp parallel num_threads(ppc::util::GetNumThreads())
  {
    std::vector<double> accumulator(A.rows_num, 0.0);
    std::vector<bool> row_mask(A.rows_num, false);
    std::vector<size_t> active_rows;

#pragma omp for schedule(dynamic)
    for (int j = 0; j < static_cast<int>(B.cols_num); ++j) {
      for (size_t kb = B.col_ptrs[j]; kb < B.col_ptrs[j + 1]; ++kb) {
        size_t k = B.row_inds[kb];
        double v_b = B.vals[kb];

        for (size_t ka = A.col_ptrs[k]; ka < A.col_ptrs[k + 1]; ++ka) {
          size_t i = A.row_inds[ka];
          if (!row_mask[i]) {
            row_mask[i] = true;
            active_rows.push_back(i);
          }
          accumulator[i] += A.vals[ka] * v_b;
        }
      }

      std::sort(active_rows.begin(), active_rows.end());
      for (size_t row_idx : active_rows) {
        if (std::abs(accumulator[row_idx]) > 1e-12) {
          temp_vals[j].push_back(accumulator[row_idx]);
          temp_rows[j].push_back(row_idx);
        }
        accumulator[row_idx] = 0.0;
        row_mask[row_idx] = false;
      }
      active_rows.clear();
    }
  }

  size_t total_nnz = 0;
  for (size_t j = 0; j < B.cols_num; ++j) {
    C.col_ptrs[j] = total_nnz;
    total_nnz += temp_vals[j].size();
  }
  C.col_ptrs[B.cols_num] = total_nnz;
  C.nnz = total_nnz;

  C.vals.reserve(total_nnz);
  C.row_inds.reserve(total_nnz);
  for (size_t j = 0; j < B.cols_num; ++j) {
    C.vals.insert(C.vals.end(), temp_vals[j].begin(), temp_vals[j].end());
    C.row_inds.insert(C.row_inds.end(), temp_rows[j].begin(), temp_rows[j].end());
  }

  return true;
}

bool RomanovMMatrixCCSOMP::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_m_matrix_ccs
