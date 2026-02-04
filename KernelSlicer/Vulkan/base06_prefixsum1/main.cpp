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
  std::vector<int> outArray2(array.size());
  
  std::srand(777);
  for(size_t i=0;i<array.size();i++) {
    array    [i] = (std::rand() % 2 == 0) ? 1 : 0;    
    outArray [i] = 0;
    outArray2[i] = 0;
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

  pImpl->CommitDeviceData();
  pImpl->PrefixSumm(array.data(), array.size(), outArray.data(), outArray2.data());

  for(int i=0;i<10;i++)
    std::cout << outArray[i] << " ";
  std::cout << std::endl;

  for(int i=0;i<10;i++)
    std::cout << outArray2[i] << " ";
  std::cout << std::endl;
  
  // check results right now
  //
  std::vector<int> refArray (outArray.size());
  std::vector<int> refArray2(outArray.size());
 
  std::exclusive_scan(array.begin(), array.end(), refArray.begin(), 0);
  std::inclusive_scan(array.begin(), array.end(), refArray2.begin(), std::plus<int>(), 0);
  
  size_t exclusiveDiffId = size_t(-1);
  size_t inclusiveDiffId = size_t(-1);

  for(size_t i=0;i<array.size();i++)
  {
    if(refArray[i] != outArray[i])
    {
      exclusiveDiffId = i;
      break;
    }
  }

  for(size_t i=0;i<array.size();i++)
  {
    if(refArray2[i] != outArray2[i])
    {
      inclusiveDiffId = i;
      break;
    }
  }
  
  if(exclusiveDiffId)
    std::cout << "[exclusive_scan]: PASSED!" << std::endl;
  else
    std::cout << "[exclusive_scan]: FAILED!" << std::endl;

  if(inclusiveDiffId)
    std::cout << "[inclusive_scan]: PASSED!" << std::endl;
  else
    std::cout << "[inclusive_scan]: FAILED!" << std::endl;

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("PrefixSumm", timings);
  std::cout << "PrefixSumm(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "PrefixSumm(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "PrefixSumm(ovrh) = " << timings[3]              << " ms " << std::endl;
  

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();  
  #endif

  return 0;
}
