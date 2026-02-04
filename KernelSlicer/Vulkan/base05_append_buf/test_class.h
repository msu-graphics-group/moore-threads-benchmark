#pragma once
#include <vector>
#include <cstdint>
#include <cstdlib>

class PrefSummTest
{
public:
  PrefSummTest(){}

  virtual void Reserve(size_t a_size) { m_values.reserve(a_size); m_values.resize(0); } 

  virtual void AppendTest(const int* a_data  [[size("a_size")]], unsigned int a_size,
                          int* a_outExc      [[size("a_size")]]);

  virtual void kernel1D_Test(const int* a_data, unsigned int a_size);                        

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  std::vector<int> m_values;
};
