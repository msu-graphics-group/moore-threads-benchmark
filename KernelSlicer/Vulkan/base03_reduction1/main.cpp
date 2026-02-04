#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstdlib>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<Numbers> CreateNumbers_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 
vk_utils::VulkanDeviceFeatures Numbers_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_CUDA
std::shared_ptr<Numbers> CreateNumbers_Generated();
#endif
#ifdef USE_ISPC
std::shared_ptr<Numbers> CreateNumbers_ISPC();
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
  bool isISPC = args.hasOption("--ispc");

  int size = 512;
  if(args.hasOption("-size"))
    size = args.getOptionValue<int>("-size");
  
  std::srand(777);
  std::vector<float> array(size*size);
  for(size_t i=0;i<array.size();i++)
    array[i] = -10.0f + 20.0f * float(std::rand()) / float(RAND_MAX);

  if(isISPC)
  {
    std::cout << "[sample04]: run ISPC ver " << std::endl;
    #ifdef USE_ISPC
    std::cout << "[OK!]     ISPC implementations exists!" << std::endl;
    #else 
    std::cout << "[FAILED!] ISPC is not defined!!!" << std::endl;
    return -1;
    #endif
  }

  std::shared_ptr<Numbers> pImpl = nullptr;
  if(onGPU)
  {
    #ifdef USE_VULKAN
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("-gpu_id", 0);
    auto features = Numbers_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features.extensionNames, enableValidationLayers, a_preferredDeviceId, &features.features2);
    pImpl         = CreateNumbers_Generated(ctx, array.size());
    #endif
    #ifdef USE_CUDA
    pImpl = CreateNumbers_Generated();
    #endif
  }
  #ifdef USE_ISPC
  else if(isISPC)
    pImpl = CreateNumbers_ISPC();
  #endif
  else
    pImpl = std::make_shared<Numbers>();
  
  pImpl->CommitDeviceData();
  
  pImpl->CalcArraySumm(array.data(), unsigned(array.size()));

  std::cout << "summ    : " <<  pImpl->m_summ << std::endl;
  std::cout << "summ_sq : " <<  pImpl->m_summSQ << std::endl;
  std::cout << "min_val : " <<  pImpl->m_minVal << std::endl;
  std::cout << "max_val : " <<  pImpl->m_maxVal << std::endl;
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("CalcArraySumm", timings);
  std::cout << "CalcArraySumm(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "CalcArraySumm(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "CalcArraySumm(ovrh) = " << timings[3]              << " ms " << std::endl;
  
  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();  
  #endif
  return 0;
}
