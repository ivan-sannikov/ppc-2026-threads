#include "zagryadskov_m_complex_spmm_ccs/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/tbb.h>

#include <util/include/util.hpp>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

ZagryadskovMComplexSpMMCCSTBB::ZagryadskovMComplexSpMMCCSTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCS();
}

LocalData::LocalData(int m, std::complex<double> zero) : marker(m, -1), acc(m, zero) {}

void ZagryadskovMComplexSpMMCCSTBB::SpMMkernel(const CCS &a, const CCS &b, const std::complex<double> &zero, double eps,
                                               tbb::enumerable_thread_specific<LocalData> &tls,
                                               std::vector<int> &col_counts) {
  tbb::parallel_for(tbb::blocked_range<int>(0, b.n), [&](const tbb::blocked_range<int> &r) {
    auto &local = tls.local();

    std::vector<int> rows;

    for (int j = r.begin(); j < r.end(); ++j) {
      rows.clear();

      for (int k = b.col_ptr[j]; k < b.col_ptr[j + 1]; ++k) {
        std::complex<double> tmpval = b.values[k];
        int btmpind = b.row_ind[k];

        for (int zp = a.col_ptr[btmpind]; zp < a.col_ptr[btmpind + 1]; ++zp) {
          int atmpind = a.row_ind[zp];
          local.acc[atmpind] += tmpval * a.values[zp];

          if (local.marker[atmpind] != j) {
            rows.push_back(atmpind);
            local.marker[atmpind] = j;
          }
        }
      }

      int count = 0;

      for (int tmpind : rows) {
        if (std::abs(local.acc[tmpind]) > eps) {
          local.values.push_back(local.acc[tmpind]);
          local.row_ind.push_back(tmpind);
          ++count;
        }
        local.acc[tmpind] = zero;
      }

      col_counts[j] = count;
    }
  });
}

void ZagryadskovMComplexSpMMCCSTBB::SpMM(const CCS &a, const CCS &b, CCS &c) {
  c.m = a.m;
  c.n = b.n;
  c.col_ptr.assign(b.n + 1, 0);
  c.row_ind.clear();
  c.values.clear();
  std::complex<double> zero(0.0, 0.0);
  const double eps = 1e-14;
  std::vector<int> col_counts(b.n, 0);

  tbb::enumerable_thread_specific<LocalData> tls([&]() { return LocalData(a.m, zero); });

  SpMMkernel(a, b, zero, eps, tls, col_counts);

  for (int j = 0; j < b.n; ++j) {
    c.col_ptr[j + 1] = c.col_ptr[j] + col_counts[j];
  }

  c.row_ind.resize(c.col_ptr[b.n]);
  c.values.resize(c.col_ptr[b.n]);

  std::vector<int> offsets = c.col_ptr;

  for (auto &local : tls) {
    int ptr = 0;
    for (int j = 0; j < b.n; ++j) {
      int cnt = col_counts[j];
      for (int k = 0; k < cnt; ++k) {
        int pos = offsets[j]++;
        c.row_ind[pos] = local.row_ind[ptr];
        c.values[pos] = local.values[ptr];
        ++ptr;
      }
    }
  }
}

bool ZagryadskovMComplexSpMMCCSTBB::ValidationImpl() {
  const CCS &a = std::get<0>(GetInput());
  const CCS &b = std::get<1>(GetInput());
  return a.n == b.m;
}

bool ZagryadskovMComplexSpMMCCSTBB::PreProcessingImpl() {
  return true;
}

bool ZagryadskovMComplexSpMMCCSTBB::RunImpl() {
  const CCS &a = std::get<0>(GetInput());
  const CCS &b = std::get<1>(GetInput());
  CCS &c = GetOutput();

  ZagryadskovMComplexSpMMCCSTBB::SpMM(a, b, c);

  return true;
}

bool ZagryadskovMComplexSpMMCCSTBB::PostProcessingImpl() {
  return true;
}

}  // namespace zagryadskov_m_complex_spmm_ccs
