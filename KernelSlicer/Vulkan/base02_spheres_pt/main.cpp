#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "test_class.h"
#include "Image2d.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<TestClass> CreateTestClass_Generated(int winWidth, int winHeight, int a_maxThreads, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  int WIN_WIDTH  = 1024;
  int WIN_HEIGHT = 1024;

  for(int i=1;i<argc;i++)
  {
    std::string strArg(argv[i]);
    if(strArg == "-size")
    {
      WIN_WIDTH  = atoi(argv[i+1]);
      WIN_HEIGHT = WIN_WIDTH;
    }
  }
  
  std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
  std::vector<uint32_t> packedXY(WIN_WIDTH*WIN_HEIGHT);
  std::vector<float4>   realColor(WIN_WIDTH*WIN_HEIGHT);

  std::shared_ptr<TestClass> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("-gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestClass_Generated(WIN_WIDTH, WIN_HEIGHT, WIN_WIDTH*WIN_HEIGHT, ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  else
    pImpl = std::make_shared<TestClass>(WIN_WIDTH, WIN_HEIGHT, WIN_WIDTH*WIN_HEIGHT);
  
  pImpl->CommitDeviceData();

  // remember pitch-linear (x,y) for each thread to make our threading 1D
  //
  pImpl->PackXYBlock(WIN_WIDTH, WIN_HEIGHT, packedXY.data(), 1);
  
  // test simple ray casting
  //
  pImpl->CastSingleRayBlock(WIN_HEIGHT*WIN_HEIGHT, packedXY.data(), pixelData.data(), 1);
  
  if(onGPU)
    LiteImage::SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    LiteImage::SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  // now test path tracing
  //
  const int PASS_NUMBER = 256;
  pImpl->StupidPathTraceBlock(WIN_HEIGHT*WIN_HEIGHT, 6, packedXY.data(), realColor.data(), PASS_NUMBER);

  const float normConst = 1.0f/float(PASS_NUMBER);
  const float invGamma  = 1.0f / 2.2f;

  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    float4 color = realColor[i]*normConst;
    color.x      = std::pow(color.x, invGamma);
    color.y      = std::pow(color.y, invGamma);
    color.z      = std::pow(color.z, invGamma);
    color.w      = 1.0f;
    pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
  }

  if(onGPU)
    LiteImage::SaveBMP("zout_gpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    LiteImage::SaveBMP("zout_cpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("StupidPathTraceBlock", timings);
  std::cout << "StupidPathTraceBlock(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "StupidPathTraceBlock(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "StupidPathTraceBlock(ovrh) = " << timings[3]              << " ms " << std::endl;

  pImpl = nullptr;  
  vk_utils::globalContextDestroy();
  return 0;
}
