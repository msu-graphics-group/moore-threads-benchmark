#pragma once
#include <vector>
#include <iostream>
#include <fstream>


class PrefSummTest
{
public:
  PrefSummTest(){}

  virtual void PrefixSumm(const int* a_data  __attribute__((size("a_size"))), unsigned int a_size,
                          int* a_outExc __attribute__((size("a_size"))), 
                          int* a_outInc __attribute__((size("a_size"))));

  virtual void kernel1D_Test(const int* a_data, unsigned int a_size);                        

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
