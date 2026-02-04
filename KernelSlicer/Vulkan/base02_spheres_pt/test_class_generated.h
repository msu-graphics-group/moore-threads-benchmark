#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <array>

#include "vk_pipeline.h"
#include "vk_buffers.h"
#include "vk_utils.h"
#include "vk_copy.h"
#include "vk_context.h"
#define TestClass_MEGAKERNEL 1

#include "test_class.h"


/////////////////////////////////////////////////////////////////////////////////////////// UBO

#include "LiteMath.h"
#ifndef CUDA_MATH
using   LiteMath::uint;
typedef LiteMath::float4x4 mat4;
typedef LiteMath::float2   vec2;
typedef LiteMath::float3   vec3;
typedef LiteMath::float4   vec4;
typedef LiteMath::int2     ivec2;
typedef LiteMath::int3     ivec3;
typedef LiteMath::int4     ivec4;
typedef LiteMath::uint2    uvec2;
typedef LiteMath::uint3    uvec3;
typedef LiteMath::uint4    uvec4;
#else
//typedef float4x4 mat4;
typedef float2   vec2;
typedef float3   vec3;
typedef float4   vec4;
typedef int2     ivec2;
typedef int3     ivec3;
typedef int4     ivec4;
typedef uint2    uvec2;
typedef uint3    uvec3;
typedef uint4    uvec4;
#endif

struct TestClass_Generated_UBO_Data
{
  mat4 m_worldViewProjInv;
  int m_height;
  int m_width;
  uint m_randomGens_capacity;
  uint m_randomGens_size;
  uint spheresMaterials_capacity;
  uint spheresMaterials_size;
  uint spheresPosRadius_capacity;
  uint spheresPosRadius_size;
  uint dummy_last;
};
/////////////////////////////////////////////////////////////////////////////////////////// UBO
class TestClass_Generated : public TestClass
{
public:

  TestClass_Generated(int winWidth, int winHeight, int a_maxThreads) : TestClass(winWidth, winHeight, a_maxThreads)
  {
  }
  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount);

  virtual void SetVulkanContext(vk_utils::VulkanContext a_ctx) { m_ctx = a_ctx; }
  virtual void SetVulkanInOutFor_StupidPathTrace(
    VkBuffer in_pakedXYBuffer,
    size_t   in_pakedXYOffset,
    VkBuffer out_colorBuffer,
    size_t   out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    StupidPathTrace_local.in_pakedXYBuffer = in_pakedXYBuffer;
    StupidPathTrace_local.in_pakedXYOffset = in_pakedXYOffset;
    StupidPathTrace_local.out_colorBuffer = out_colorBuffer;
    StupidPathTrace_local.out_colorOffset = out_colorOffset;
    UpdateAllGeneratedDescriptorSets_StupidPathTrace();
  }

  virtual void SetVulkanInOutFor_CastSingleRay(
    VkBuffer in_pakedXYBuffer,
    size_t   in_pakedXYOffset,
    VkBuffer out_colorBuffer,
    size_t   out_colorOffset,
    uint32_t dummyArgument = 0)
  {
    CastSingleRay_local.in_pakedXYBuffer = in_pakedXYBuffer;
    CastSingleRay_local.in_pakedXYOffset = in_pakedXYOffset;
    CastSingleRay_local.out_colorBuffer = out_colorBuffer;
    CastSingleRay_local.out_colorOffset = out_colorOffset;
    UpdateAllGeneratedDescriptorSets_CastSingleRay();
  }

  virtual void SetVulkanInOutFor_PackXY(
    VkBuffer out_pakedXYBuffer,
    size_t   out_pakedXYOffset,
    uint32_t dummyArgument = 0)
  {
    PackXY_local.out_pakedXYBuffer = out_pakedXYBuffer;
    PackXY_local.out_pakedXYOffset = out_pakedXYOffset;
    UpdateAllGeneratedDescriptorSets_PackXY();
  }

  virtual ~TestClass_Generated();


virtual void InitDeviceData() ;
virtual void UpdateDeviceData(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine)   {
    UpdateVectorMembers(a_pCopyEngine);
    UpdateTextureMembers(a_pCopyEngine);
    UpdatePlainMembers(a_pCopyEngine);
  }

  std::shared_ptr<vk_utils::ICopyEngine> m_pLastCopyHelper = nullptr;
  virtual void DeleteDeviceData();
  virtual void CommitDeviceData(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper) // you have to define this virtual function in the original imput class
  {
    ReserveEmptyVectors();
    DeleteDeviceData();
    InitDeviceData();
    UpdateDeviceData(a_pCopyHelper);
    m_pLastCopyHelper = a_pCopyHelper;
    m_commitCount++;
  }
  void CommitDeviceData() override { CommitDeviceData(m_ctx.pCopyHelper); }
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override;
  
  virtual void ReserveEmptyVectors();
  virtual void UpdatePlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateVectorMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void UpdateTextureMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  virtual void ReadPlainMembers(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine);
  static VkPhysicalDeviceFeatures2 ListRequiredDeviceFeatures(std::vector<const char*>& deviceExtensions);

  virtual void StupidPathTraceCmd(VkCommandBuffer a_commandBuffer, uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color);
  virtual void CastSingleRayCmd(VkCommandBuffer a_commandBuffer, uint tid, const uint* in_pakedXY, uint* out_color);
  virtual void PackXYCmd(VkCommandBuffer a_commandBuffer, uint tidX, uint tidY, uint* out_pakedXY);

  void StupidPathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint32_t a_numPasses) override;
  void CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint32_t a_numPasses) override;
  void PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint32_t a_numPasses) override;

  inline vk_utils::ExecTime GetStupidPathTraceExecutionTime() const { return m_exTimeStupidPathTrace; }
  inline vk_utils::ExecTime GetCastSingleRayExecutionTime() const { return m_exTimeCastSingleRay; }
  inline vk_utils::ExecTime GetPackXYExecutionTime() const { return m_exTimePackXY; }

  vk_utils::ExecTime m_exTimeStupidPathTrace;
  vk_utils::ExecTime m_exTimeCastSingleRay;
  vk_utils::ExecTime m_exTimePackXY;

  virtual void copyKernelFloatCmd(uint32_t length);
  virtual void matMulTransposeCmd(uint32_t A_offset, uint32_t B_offset, uint32_t C_offset, uint32_t A_col_len, uint32_t B_col_len, uint32_t A_row_len);

  virtual void StupidPathTraceMegaCmd(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color);
  virtual void CastSingleRayMegaCmd(uint tid, const uint* in_pakedXY, uint* out_color);
  virtual void PackXYMegaCmd(uint tidX, uint tidY, uint* out_pakedXY);

  struct MemLoc
  {
    VkDeviceMemory memObject = VK_NULL_HANDLE;
    size_t         memOffset = 0;
    size_t         allocId   = 0;
  };

  virtual MemLoc AllocAndBind(const std::vector<VkBuffer>& a_buffers, VkMemoryAllocateFlags a_flags = 0); ///< replace this function to apply custom allocator
  virtual MemLoc AllocAndBind(const std::vector<VkImage>& a_image,    VkMemoryAllocateFlags a_flags = 0);    ///< replace this function to apply custom allocator
  virtual void   FreeAllAllocations(std::vector<MemLoc>& a_memLoc);    ///< replace this function to apply custom allocator

protected:

  VkPhysicalDevice           m_physicalDevice = VK_NULL_HANDLE;
  VkDevice                   m_device         = VK_NULL_HANDLE;
  vk_utils::VulkanContext    m_ctx          = {};
  VkCommandBuffer            m_currCmdBuffer   = VK_NULL_HANDLE;
  uint32_t                   m_currThreadFlags = 0;
  std::vector<MemLoc>        m_allMems;
  VkPhysicalDeviceProperties m_devProps;
  size_t                     m_commitCount = 0;

  VkBufferMemoryBarrier BarrierForClearFlags(VkBuffer a_buffer);
  VkBufferMemoryBarrier BarrierForSingleBuffer(VkBuffer a_buffer);
  void BarriersForSeveralBuffers(VkBuffer* a_inBuffers, VkBufferMemoryBarrier* a_outBarriers, uint32_t a_buffersNum);

  virtual void InitHelpers();
  virtual void InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay = true);
  virtual void InitKernels(const char* a_filePath);
  virtual void AllocateAllDescriptorSets();

  virtual void UpdateAllGeneratedDescriptorSets_StupidPathTrace();
  virtual void UpdateAllGeneratedDescriptorSets_CastSingleRay();
  virtual void UpdateAllGeneratedDescriptorSets_PackXY();

  virtual void AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem);

  virtual void AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_image);
  virtual std::string AlterShaderPath(const char* in_shaderPath) { return std::string("") + std::string(in_shaderPath); }
  
  

  struct StupidPathTrace_Data
  {
    VkBuffer in_pakedXYBuffer = VK_NULL_HANDLE;
    size_t   in_pakedXYOffset = 0;
    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;
    bool needToClearOutput = true;
  } StupidPathTrace_local;

  struct CastSingleRay_Data
  {
    VkBuffer in_pakedXYBuffer = VK_NULL_HANDLE;
    size_t   in_pakedXYOffset = 0;
    VkBuffer out_colorBuffer = VK_NULL_HANDLE;
    size_t   out_colorOffset = 0;
    bool needToClearOutput = true;
  } CastSingleRay_local;

  struct PackXY_Data
  {
    VkBuffer out_pakedXYBuffer = VK_NULL_HANDLE;
    size_t   out_pakedXYOffset = 0;
    bool needToClearOutput = true;
  } PackXY_local;



  struct MembersDataGPU
  {
    VkBuffer m_randomGensBuffer = VK_NULL_HANDLE;
    size_t   m_randomGensOffset = 0;
    VkBuffer spheresMaterialsBuffer = VK_NULL_HANDLE;
    size_t   spheresMaterialsOffset = 0;
    VkBuffer spheresPosRadiusBuffer = VK_NULL_HANDLE;
    size_t   spheresPosRadiusOffset = 0;
  } m_vdata;


  size_t m_maxThreadCount = 0;
  VkBuffer m_classDataBuffer = VK_NULL_HANDLE;

  VkPipelineLayout      StupidPathTraceMegaLayout   = VK_NULL_HANDLE;
  VkPipeline            StupidPathTraceMegaPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout StupidPathTraceMegaDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateStupidPathTraceMegaDSLayout();
  virtual void InitKernel_StupidPathTraceMega(const char* a_filePath);
  VkPipelineLayout      CastSingleRayMegaLayout   = VK_NULL_HANDLE;
  VkPipeline            CastSingleRayMegaPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout CastSingleRayMegaDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreateCastSingleRayMegaDSLayout();
  virtual void InitKernel_CastSingleRayMega(const char* a_filePath);
  VkPipelineLayout      PackXYMegaLayout   = VK_NULL_HANDLE;
  VkPipeline            PackXYMegaPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout PackXYMegaDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatePackXYMegaDSLayout();
  virtual void InitKernel_PackXYMega(const char* a_filePath);
  VkPipelineCache m_pipelineCache = VK_NULL_HANDLE; // if pipeline cache not enabled, will be VK_NULL_HANDLE 
  virtual VkBufferUsageFlags GetAdditionalFlagsForUBO() const;
  virtual uint32_t           GetDefaultMaxTextures() const;

  VkPipelineLayout      copyKernelFloatLayout   = VK_NULL_HANDLE;
  VkPipeline            copyKernelFloatPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyKernelFloatDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatecopyKernelFloatDSLayout();

  VkPipelineLayout      matMulTransposeLayout   = VK_NULL_HANDLE;
  VkPipeline            matMulTransposePipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout matMulTransposeDSLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout CreatematMulTransposeDSLayout();

  VkDescriptorPool m_dsPool = VK_NULL_HANDLE;
  VkDescriptorSet  m_allGeneratedDS[3];

  TestClass_Generated_UBO_Data m_uboData;

  constexpr static uint32_t MEMCPY_BLOCK_SIZE = 256;
  constexpr static uint32_t REDUCTION_BLOCK_SIZE = 256;

  virtual void SceneRestrictions(uint32_t a_restrictions[4]) const
  {
    uint32_t maxMeshes            = 1024;
    uint32_t maxTotalVertices     = 1'000'000;
    uint32_t maxTotalPrimitives   = 1'000'000;
    uint32_t maxPrimitivesPerMesh = 200'000;

    a_restrictions[0] = maxMeshes;
    a_restrictions[1] = maxTotalVertices;
    a_restrictions[2] = maxTotalPrimitives;
    a_restrictions[3] = maxPrimitivesPerMesh;
  }
  virtual void MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout,
                                            VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline);
  virtual void MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout,
                                       VkPipeline* pPipeline);

  std::vector<VkPipelineLayout> m_allCreatedPipelineLayouts; ///<! remenber them here to delete later
  std::vector<VkPipeline>       m_allCreatedPipelines;       ///<! remenber them here to delete later
public:

  struct MegaKernelIsEnabled
  {
    bool enableStupidPathTraceMega = true;
    bool enableCastSingleRayMega = true;
    bool enablePackXYMega = true;
    bool dummy = 0;
  };

  static MegaKernelIsEnabled  m_megaKernelFlags;
  static MegaKernelIsEnabled& EnabledPipelines() { return m_megaKernelFlags; }
  uint32_t m_subgroupSize = 1; 
};


