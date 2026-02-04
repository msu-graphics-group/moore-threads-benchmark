#include "test_class.h"
#include <string>
#include <chrono>

void Numbers::CalcArraySumm(const float* a_data, unsigned int a_dataSize1)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel1D_ArraySumm(a_data, a_dataSize1);
  m_executionTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}

void Numbers::kernel1D_ArraySumm(const float* a_data, unsigned int a_dataSize)
{
  for(int i=0; i<a_dataSize; i++)
  {
    const float val = a_data[i];

    m_summ   += val;
    m_summSQ += val*val;
    m_minVal = std::min(m_minVal, val);
    m_maxVal = std::max(m_maxVal, val);
  }
}