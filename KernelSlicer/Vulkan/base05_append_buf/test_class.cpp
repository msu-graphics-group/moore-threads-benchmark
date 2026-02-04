#include "test_class.h"


void PrefSummTest::AppendTest(const int* a_data, unsigned int a_size, int* a_outExc)
{
  kernel1D_Test(a_data, a_size); // need to detect 'PrefixSumm' as control function
}

void PrefSummTest::kernel1D_Test(const int* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    const int val = a_data[i];
    if(val != 0)
      m_values.push_back(val);
  }
}