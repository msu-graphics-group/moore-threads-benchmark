#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <numeric>

#include <cstdint>
#include <cstdlib>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<PrefSummTest> CreatePrefSummTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  bool onGPU  = args.hasOption("--gpu");

  int size = 512;
  if(args.hasOption("-size"))
    size = args.getOptionValue<int>("-size");

  std::vector<int> array    (size*size); //
  std::vector<int> outArray (array.size());
  
  std::srand(777);
  for(size_t i=0;i<array.size();i++) {
    array    [i] = (std::rand() % 2 == 0) ? 1 : 0;    
    outArray [i] = 0;
  }

  std::shared_ptr<PrefSummTest> pImpl = nullptr;

  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("-gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreatePrefSummTest_Generated(ctx, array.size());
  }
  else
  #endif
    pImpl = std::make_shared<PrefSummTest>();
  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->Reserve(array.size());

  pImpl->CommitDeviceData();
  pImpl->AppendTest(array.data(), array.size(), outArray.data());

  for(int i=0;i<10;i++)
    std::cout << outArray[i] << " ";
  std::cout << std::endl;
  
  std::cout << "Numbers Appended: " << pImpl->m_values.size() << std::endl;

  // check results right now

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("AppendTest", timings);
  
  std::cout << "AppendTest(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "AppendTest(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "AppendTest(ovrh) = " << timings[3]              << " ms " << std::endl;
  
  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();  
  #endif

  return 0;
}
