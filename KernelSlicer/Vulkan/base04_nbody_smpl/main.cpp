#include <iostream>
#include <memory>

#include "test_class.h"
#include "ArgParser.h"

#ifdef USE_VULKAN
#include "vk_context.h"
vk_utils::VulkanDeviceFeatures nBody_Generated_ListRequiredDeviceFeatures();
std::shared_ptr<nBody> CreatenBody_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
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

  int size = 256;
  if(args.hasOption("-size"))
    size = args.getOptionValue<int>("-size");

  std::vector<nBody::BodyState> outBodies(size*size);

  std::shared_ptr<nBody> pImpl = nullptr;

  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("-gpu_id", 0);
    auto features = nBody_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features.extensionNames, enableValidationLayers, a_preferredDeviceId, &features.features2);
    pImpl         = CreatenBody_Generated(ctx, outBodies.size());
  }
  else
  #endif
  pImpl = std::make_shared<nBody>();
  
  pImpl->setParameters(outBodies.size(), 777, 1);
  pImpl->CommitDeviceData();

  pImpl->perform(outBodies.data(), uint32_t(outBodies.size()));
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("perform", timings);
  std::cout << "perform(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "perform(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "perform(ovrh) = " << timings[3]              << " ms " << std::endl;
  
  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();  
  #endif
  return 0;
}
