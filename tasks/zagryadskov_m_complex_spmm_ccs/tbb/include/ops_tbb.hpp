#pragma once

#include <tbb/enumerable_thread_specific.h>

#include <complex>
#include <vector>

#include "task/include/task.hpp"
#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

struct LocalData {
  std::vector<int> row_ind;
  std::vector<std::complex<double>> values;
  std::vector<int> col_ptr;

  std::vector<int> marker;
  std::vector<std::complex<double>> acc;

  LocalData(int m, std::complex<double> zero);
};

class ZagryadskovMComplexSpMMCCSTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }
  explicit ZagryadskovMComplexSpMMCCSTBB(const InType &in);

 private:
  static void SpMM(const CCS &a, const CCS &b, CCS &c);
  static void SpMMkernel(const CCS &a, const CCS &b, const std::complex<double> &zero, double eps,
                         tbb::enumerable_thread_specific<LocalData> &tls, std::vector<int> &col_counts);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zagryadskov_m_complex_spmm_ccs
