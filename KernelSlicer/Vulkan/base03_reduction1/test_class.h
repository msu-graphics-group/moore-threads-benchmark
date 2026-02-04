#pragma once
#include <vector>
#include <iostream>
#include <fstream>

class Numbers
{
public:
  Numbers(){ m_summ = 0.0f; m_summSQ = 0.0f; m_minVal = 1e20f; m_maxVal = -1e20f; }

  virtual void CalcArraySumm(const float* a_data [[size("a_dataSize")]], unsigned int a_dataSize1);
  virtual void kernel1D_ArraySumm(const float* a_data, unsigned int a_dataSize);
  
  float m_summ;
  float m_summSQ;
  float m_minVal;
  float m_maxVal;

  virtual void CommitDeviceData() {}                                                                    // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_executionTime; } // will be overriden in generated class
  float m_executionTime = 0.0f;
};
