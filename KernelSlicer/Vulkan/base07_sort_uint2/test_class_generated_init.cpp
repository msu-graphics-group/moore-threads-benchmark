#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <cassert>
#include "vk_copy.h"
#include "vk_context.h"
#include "test_class_generated.h"


std::shared_ptr<Sorter> CreateSorter_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<Sorter_Generated>();
  pObj->SetVulkanContext(a_ctx);
  pObj->InitVulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}

vk_utils::VulkanDeviceFeatures Sorter_Generated_ListRequiredDeviceFeatures()
{
  vk_utils::VulkanDeviceFeatures res;
  res.features2 = Sorter_Generated::ListRequiredDeviceFeatures(res.extensionNames);
  res.apiVersion = VK_API_VERSION_1_1;
  return res;
}

void Sorter_Generated::InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount)
{
  m_physicalDevice = a_physicalDevice;
  m_device         = a_device;
  m_allCreatedPipelineLayouts.reserve(256);
  m_allCreatedPipelines.reserve(256);
  InitHelpers();
  InitBuffers(a_maxThreadsCount, true);
  InitKernels(".spv");
  AllocateAllDescriptorSets();
  // get timestampPeriod from device props
  //
  VkPhysicalDeviceProperties2 physicalDeviceProperties{};
  VkPhysicalDeviceSubgroupProperties  subgroupProperties{};
  subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  subgroupProperties.pNext = nullptr;
  physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  physicalDeviceProperties.pNext = &subgroupProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &physicalDeviceProperties);
  m_subgroupSize    = subgroupProperties.subgroupSize;
}

static uint32_t ComputeReductionAuxBufferElements(uint32_t whole_size, uint32_t wg_size)
{
  uint32_t sizeTotal = 0;
  while (whole_size > 1)
  {
    whole_size  = (whole_size + wg_size - 1) / wg_size;
    sizeTotal  += std::max<uint32_t>(whole_size, 1);
  }
  return sizeTotal;
}

VkBufferUsageFlags Sorter_Generated::GetAdditionalFlagsForUBO() const
{
  return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
}

uint32_t Sorter_Generated::GetDefaultMaxTextures() const { return 256; }

void Sorter_Generated::MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline)
{
  VkPipelineShaderStageCreateInfo shaderStageInfo = {};
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  auto shaderCode   = vk_utils::readSPVFile(a_shaderPath);
  auto shaderModule = vk_utils::createShaderModule(m_device, shaderCode);

  shaderStageInfo.module              = shaderModule;
  shaderStageInfo.pName               = a_mainName;
  shaderStageInfo.pSpecializationInfo = a_specInfo;

  VkPushConstantRange pcRange = {};
  pcRange.stageFlags = shaderStageInfo.stage;
  pcRange.offset     = 0;
  pcRange.size       = 128; // at least 128 bytes for push constants for all Vulkan implementations

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pcRange;
  pipelineLayoutInfo.pSetLayouts            = &a_dsLayout;
  pipelineLayoutInfo.setLayoutCount         = 1;

  VkResult res = vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, pPipelineLayout);
  if(res != VK_SUCCESS)
  {
    std::string errMsg = vk_utils::errorString(res);
    std::cout << "[ShaderError]: vkCreatePipelineLayout have failed for '" << a_shaderPath << "' with '" << errMsg.c_str() << "'" << std::endl;
  }
  else
    m_allCreatedPipelineLayouts.push_back(*pPipelineLayout);

  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.flags              = 0;
  pipelineInfo.stage              = shaderStageInfo;
  pipelineInfo.layout             = (*pPipelineLayout);
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  res = vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, pPipeline);
  if(res != VK_SUCCESS)
  {
    std::string errMsg = vk_utils::errorString(res);
    std::cout << "[ShaderError]: vkCreateComputePipelines have failed for '" << a_shaderPath << "' with '" << errMsg.c_str() << "'" << std::endl;
  }
  else
    m_allCreatedPipelines.push_back(*pPipeline);

  if (shaderModule != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_device, shaderModule, VK_NULL_HANDLE);
}

void Sorter_Generated::MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout, VkPipeline* pPipeline)
{
  VkPipelineShaderStageCreateInfo shaderStageInfo = {};
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  auto shaderCode   = vk_utils::readSPVFile(a_shaderPath);
  auto shaderModule = vk_utils::createShaderModule(m_device, shaderCode);

  shaderStageInfo.module              = shaderModule;
  shaderStageInfo.pName               = a_mainName;
  shaderStageInfo.pSpecializationInfo = a_specInfo;

  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.flags              = 0;
  pipelineInfo.stage              = shaderStageInfo;
  pipelineInfo.layout             = pipelineLayout;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  VkResult res = vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, pPipeline);
  if(res != VK_SUCCESS)
  {
    std::string errMsg = vk_utils::errorString(res);
    std::cout << "[ShaderError]: vkCreateComputePipelines have failed for '" << a_shaderPath << "' with '" << errMsg.c_str() << "'" << std::endl;
  }
  else
    m_allCreatedPipelines.push_back(*pPipeline);

  if (shaderModule != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_device, shaderModule, VK_NULL_HANDLE);
}

void Sorter_Generated::DeleteDeviceData()
{
  if(m_commitCount == 0)
    return;
  vkDestroyBuffer(m_device, m_classDataBuffer, nullptr);
  if(copyKernelFloatDSLayout != VK_NULL_HANDLE)
     vkDestroyDescriptorSetLayout(m_device, copyKernelFloatDSLayout, nullptr);
  FreeAllAllocations(m_allMems);
}

Sorter_Generated::~Sorter_Generated()
{
  for(size_t i=0;i<m_allCreatedPipelines.size();i++)
    vkDestroyPipeline(m_device, m_allCreatedPipelines[i], nullptr);
  for(size_t i=0;i<m_allCreatedPipelineLayouts.size();i++)
    vkDestroyPipelineLayout(m_device, m_allCreatedPipelineLayouts[i], nullptr);
  m_sort_uvec2.DeleteDSLayouts(m_device);
  vkDestroyDescriptorSetLayout(m_device, CopyDataDSLayout, nullptr);
  CopyDataDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorPool(m_device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;
  DeleteDeviceData();
}

void Sorter_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(m_physicalDevice, &m_devProps);
}


void Sorter_Generated::InitKernel_CopyData(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_generated/kernel1D_CopyData.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  CopyDataDSLayout = CreateCopyDataDSLayout();
  if(true)
  {
    MakeComputePipelineAndLayout(shaderPath.c_str(), "main", kspec, CopyDataDSLayout, &CopyDataLayout, &CopyDataPipeline);
  }
  else
  {
    CopyDataLayout   = nullptr;
    CopyDataPipeline = nullptr;
  }
}


void Sorter_Generated::InitKernels(const char* a_filePath)
{
  InitKernel_CopyData(a_filePath);
  // init m_sort_uvec2
  {
    std::string bitonicPassPath = AlterShaderPath("shaders_generated/z_bitonic_uvec2_pass.comp.spv");
    std::string bitonic512Path  = AlterShaderPath("shaders_generated/z_bitonic_uvec2_512.comp.spv");
    std::string bitonic1024Path = AlterShaderPath("shaders_generated/z_bitonic_uvec2_1024.comp.spv");
    std::string bitonic2048Path = AlterShaderPath("shaders_generated/z_bitonic_uvec2_2048.comp.spv");

    m_sort_uvec2.sortDSLayout = m_sort_uvec2.CreateSortDSLayout(m_device);
    MakeComputePipelineAndLayout(bitonicPassPath.c_str(),  "main", nullptr, m_sort_uvec2.sortDSLayout, &m_sort_uvec2.bitonicPassLayout, &m_sort_uvec2.bitonicPassPipeline);

    if(m_devProps.limits.maxComputeWorkGroupSize[0] >= 256)
    {
      MakeComputePipelineAndLayout(bitonic512Path.c_str(), "main", nullptr, m_sort_uvec2.sortDSLayout, &m_sort_uvec2.bitonic512Layout, &m_sort_uvec2.bitonic512Pipeline);
    }
    else
    {
      m_sort_uvec2.bitonic512Layout   = VK_NULL_HANDLE;
      m_sort_uvec2.bitonic512Pipeline = VK_NULL_HANDLE;
    }

    if(m_devProps.limits.maxComputeWorkGroupSize[0] >= 512)
    {
      MakeComputePipelineAndLayout(bitonic1024Path.c_str(), "main", nullptr, m_sort_uvec2.sortDSLayout, &m_sort_uvec2.bitonic1024Layout, &m_sort_uvec2.bitonic1024Pipeline);
    }
    else
    {
      m_sort_uvec2.bitonic1024Layout   = VK_NULL_HANDLE;
      m_sort_uvec2.bitonic1024Pipeline = VK_NULL_HANDLE;
    }

    if(m_devProps.limits.maxComputeWorkGroupSize[0] >= 1024)
    {
      MakeComputePipelineAndLayout(bitonic2048Path.c_str(), "main", nullptr, m_sort_uvec2.sortDSLayout, &m_sort_uvec2.bitonic2048Layout, &m_sort_uvec2.bitonic2048Pipeline);
    }
    else
    {
      m_sort_uvec2.bitonic2048Layout   = VK_NULL_HANDLE;
      m_sort_uvec2.bitonic2048Pipeline = VK_NULL_HANDLE;
    }
  }
}

void Sorter_Generated::InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay)
{
  ReserveEmptyVectors();

  m_maxThreadCount = a_maxThreadsCount;
  std::vector<VkBuffer> allBuffers;
  allBuffers.reserve(64);

  struct BufferReqPair
  {
    BufferReqPair() {  }
    BufferReqPair(VkBuffer a_buff, VkDevice a_dev) : buf(a_buff) { vkGetBufferMemoryRequirements(a_dev, a_buff, &req); }
    VkBuffer             buf = VK_NULL_HANDLE;
    VkMemoryRequirements req = {};
  };

  struct LocalBuffers
  {
    std::vector<BufferReqPair> bufs;
    size_t                     size = 0;
    std::vector<VkBuffer>      bufsClean;
  };

  std::vector<LocalBuffers> groups;
  groups.reserve(16);


  size_t largestIndex = 0;
  size_t largestSize  = 0;
  for(size_t i=0;i<groups.size();i++)
  {
    if(groups[i].size > largestSize)
    {
      largestIndex = i;
      largestSize  = groups[i].size;
    }
    groups[i].bufsClean.resize(groups[i].bufs.size());
    for(size_t j=0;j<groups[i].bufsClean.size();j++)
      groups[i].bufsClean[j] = groups[i].bufs[j].buf;
  }
  auto& allBuffersRef = allBuffers;

  auto internalBuffersMem = AllocAndBind(allBuffersRef);
  if(a_tempBuffersOverlay)
  {
    for(size_t i=0;i<groups.size();i++)
      if(i != largestIndex)
        AssignBuffersToMemory(groups[i].bufsClean, internalBuffersMem.memObject);
  }
}

void Sorter_Generated::ReserveEmptyVectors()
{
}

void Sorter_Generated::InitDeviceData()
{
  std::vector<VkBuffer> memberVectorsWithDevAddr;
  std::vector<VkBuffer> memberVectors;
  std::vector<VkImage>  memberTextures;
  m_classDataBuffer = vk_utils::createBuffer(m_device, sizeof(m_uboData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | GetAdditionalFlagsForUBO() | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  memberVectors.push_back(m_classDataBuffer);
  


  AllocMemoryForMemberBuffersAndImages(memberVectors, memberTextures);
  if(memberVectorsWithDevAddr.size() != 0)
    AllocAndBind(memberVectorsWithDevAddr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
}




void Sorter_Generated::AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem)
{
  if(a_buffers.size() == 0 || a_mem == VK_NULL_HANDLE)
    return;

  std::vector<VkMemoryRequirements> memInfos(a_buffers.size());
  for(size_t i=0;i<memInfos.size();i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkGetBufferMemoryRequirements(m_device, a_buffers[i], &memInfos[i]);
    else
    {
      memInfos[i] = memInfos[0];
      memInfos[i].size = 0;
    }
  }

  for(size_t i=1;i<memInfos.size();i++)
  {
    if(memInfos[i].memoryTypeBits != memInfos[0].memoryTypeBits)
    {
      std::cout << "[Sorter_Generated::AssignBuffersToMemory]: error, input buffers has different 'memReq.memoryTypeBits'" << std::endl;
      return;
    }
  }

  auto offsets = vk_utils::calculateMemOffsets(memInfos);
  for (size_t i = 0; i < memInfos.size(); i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkBindBufferMemory(m_device, a_buffers[i], a_mem, offsets[i]);
  }
}

Sorter_Generated::MemLoc Sorter_Generated::AllocAndBind(const std::vector<VkBuffer>& a_buffers, VkMemoryAllocateFlags a_flags)
{
  MemLoc currLoc;
  if(a_buffers.size() > 0)
  {
    currLoc.memObject = vk_utils::allocateAndBindWithPadding(m_device, m_physicalDevice, a_buffers, a_flags);
    currLoc.allocId   = m_allMems.size();
    m_allMems.push_back(currLoc);
  }
  return currLoc;
}

Sorter_Generated::MemLoc Sorter_Generated::AllocAndBind(const std::vector<VkImage>& a_images, VkMemoryAllocateFlags a_flags)
{
  MemLoc currLoc;
  if(a_images.size() > 0)
  {
    std::vector<VkMemoryRequirements> reqs(a_images.size());
    for(size_t i=0; i<reqs.size(); i++)
      vkGetImageMemoryRequirements(m_device, a_images[i], &reqs[i]);

    for(size_t i=0; i<reqs.size(); i++)
    {
      if(reqs[i].memoryTypeBits != reqs[0].memoryTypeBits)
      {
        std::cout << "Sorter_Generated::AllocAndBind(textures): memoryTypeBits warning, need to split mem allocation (override me)" << std::endl;
        break;
      }
    }

    auto offsets  = vk_utils::calculateMemOffsets(reqs);
    auto memTotal = offsets[offsets.size() - 1];

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = memTotal;
    allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(reqs[0].memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_physicalDevice);
    VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocateInfo, NULL, &currLoc.memObject));

    for(size_t i=0;i<a_images.size();i++) {
      VK_CHECK_RESULT(vkBindImageMemory(m_device, a_images[i], currLoc.memObject, offsets[i]));
    }

    currLoc.allocId = m_allMems.size();
    m_allMems.push_back(currLoc);
  }
  return currLoc;
}

void Sorter_Generated::FreeAllAllocations(std::vector<MemLoc>& a_memLoc)
{
  // in general you may check 'mem.allocId' for unique to be sure you dont free mem twice
  // for default implementation this is not needed
  for(auto mem : a_memLoc)
    vkFreeMemory(m_device, mem.memObject, nullptr);
  a_memLoc.resize(0);
}

void Sorter_Generated::AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_images)
{
  std::vector<VkMemoryRequirements> bufMemReqs(a_buffers.size()); // we must check that all buffers have same memoryTypeBits;
  for(size_t i = 0; i < a_buffers.size(); ++i)                    // if not, split to multiple allocations
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkGetBufferMemoryRequirements(m_device, a_buffers[i], &bufMemReqs[i]);
    else
    {
      bufMemReqs[i] = bufMemReqs[0];
      bufMemReqs[i].size = 0;
    }
  }

  bool needSplit = false;
  for(size_t i = 1; i < bufMemReqs.size(); ++i)
  {
    if(bufMemReqs[i].memoryTypeBits != bufMemReqs[0].memoryTypeBits)
    {
      needSplit = true;
      break;
    }
  }

  if(needSplit)
  {
    std::unordered_map<uint32_t, std::vector<uint32_t> > bufferSets;
    for(uint32_t j = 0; j < uint32_t(bufMemReqs.size()); ++j)
    {
      uint32_t key = uint32_t(bufMemReqs[j].memoryTypeBits);
      bufferSets[key].push_back(j);
    }

    for(const auto& buffGroup : bufferSets)
    {
      std::vector<VkBuffer> currGroup;
      for(auto id : buffGroup.second)
        currGroup.push_back(a_buffers[id]);
      AllocAndBind(currGroup);
    }
  }
  else
    AllocAndBind(a_buffers);

}
VkDescriptorSetLayout Sorter_Generated::BitonicSortData::CreateSortDSLayout(VkDevice a_device)
{
  std::array<VkDescriptorSetLayoutBinding, 1> dsBindings;

  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = dsBindings.size();
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

void Sorter_Generated::BitonicSortData::DeleteDSLayouts(VkDevice a_device)
{
  vkDestroyDescriptorSetLayout(a_device, sortDSLayout, nullptr);
}

static bool isPowerOfTwo(size_t n)
{
  if (n == 0)
    return false;
  return (std::ceil(std::log2(n)) == std::floor(std::log2(n)));
}

void Sorter_Generated::BitonicSortData::BitonicSortSimpleCmd(VkCommandBuffer a_cmdBuffer, size_t a_size)
{
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };

  if(!isPowerOfTwo(a_size))
  {
    std::cout << "BitonicSortCmd, bad input size " << a_size << ", it must be power of 2" << std::endl;
    return;
  }

  int numStages = 0;
  for (size_t temp = a_size; temp > 2; temp >>= 1)
    numStages++;

  const size_t blockSizeX = 256;
  struct KernelArgsPC
  {
    int iNumElementsX;
    int stage;
    int passOfStage;
    int invertModeOn;
  } pcData;

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonicPassPipeline);

  // up, form bitonic sequence with half arrays
  //
  for (int stage = 0; stage < numStages; stage++)
  {
    for (int passOfStage = stage; passOfStage >= 0; passOfStage--)
    {
      pcData.iNumElementsX = int(a_size);
      pcData.stage         = stage;
      pcData.passOfStage   = passOfStage;
      pcData.invertModeOn  = 1;
      vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
      vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    }
  }

  // down, finally sort it
  //
  for(int passOfStage = numStages; passOfStage >= 0; passOfStage--)
  {
    pcData.iNumElementsX = int(a_size);
    pcData.stage         = numStages - 1;
    pcData.passOfStage   = passOfStage;
    pcData.invertModeOn  = 0;
    vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
    vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
  }
}

void Sorter_Generated::BitonicSortData::BitonicSortCmd(VkCommandBuffer a_cmdBuffer, size_t a_size, uint32_t a_maxWorkGroupSize)
{
  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };

  if(!isPowerOfTwo(a_size))
  {
    std::cout << "BitonicSortCmd, bad input size " << a_size << ", it must be power of 2" << std::endl;
    return;
  }

  int numStages = 0;
  for (size_t temp = a_size; temp > 2; temp >>= 1)
    numStages++;

  const size_t blockSizeX = 256;
  struct KernelArgsPC
  {
    int iNumElementsX;
    int stage;
    int passOfStage;
    int invertModeOn;
  } pcData;

  // up, form bitonic sequence with half arrays
  //
  for (int stage = 0; stage < numStages; stage++)
  {
    vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonicPassPipeline);

    for (int passOfStage = stage; passOfStage >= 0; passOfStage--)
    {
      bool stopNow = false;

      if (passOfStage > 0 && passOfStage <= 10 && a_maxWorkGroupSize >= 1024)
      {
        vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic2048Pipeline);
        stopNow = true;
      }
      else if (passOfStage > 0 && passOfStage <= 9 && a_maxWorkGroupSize >= 512)
      {
        vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic1024Pipeline);
        stopNow = true;
      }
      else if (passOfStage > 0 && passOfStage <= 8 && a_maxWorkGroupSize >= 256)
      {
        vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic512Pipeline);
        stopNow = true;
      }

      pcData.iNumElementsX = int(a_size);
      pcData.stage         = stage;
      pcData.passOfStage   = passOfStage;
      pcData.invertModeOn  = 1;
      vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
      vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

      if(stopNow)
        break;
    }
  }

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonicPassPipeline);

  // down, finally sort it
  //
  for(int passOfStage = numStages; passOfStage >= 0; passOfStage--)
  {
    bool stopNow = false;

    if (passOfStage > 0 && passOfStage <= 10 && a_maxWorkGroupSize >= 1024)
    {
      vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic2048Pipeline);
      stopNow = true;
    }
    else if (passOfStage > 0 && passOfStage <= 9 && a_maxWorkGroupSize >= 512)
    {
      vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic1024Pipeline);
      stopNow = true;
    }
    else if (passOfStage > 0 && passOfStage <= 8 && a_maxWorkGroupSize >= 256)
    {
      vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bitonic512Pipeline);
      stopNow = true;
    }

    pcData.iNumElementsX = int(a_size);
    pcData.stage         = numStages - 1;
    pcData.passOfStage   = passOfStage;
    pcData.invertModeOn  = 0;
    vkCmdPushConstants(a_cmdBuffer, bitonicPassLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (a_size + blockSizeX - 1) / blockSizeX, 1, 1);
    vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    if(stopNow)
      break;
  }
}

VkPhysicalDeviceFeatures2 Sorter_Generated::ListRequiredDeviceFeatures(std::vector<const char*>& deviceExtensions)
{
  static VkPhysicalDeviceFeatures2 features2 = {};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = nullptr;
  features2.features.shaderInt64   = false;
  features2.features.shaderFloat64 = false;
  features2.features.shaderInt16   = false;
  
  void** ppNext = &features2.pNext;
  return features2;
}

Sorter_Generated::MegaKernelIsEnabled Sorter_Generated::m_megaKernelFlags;

