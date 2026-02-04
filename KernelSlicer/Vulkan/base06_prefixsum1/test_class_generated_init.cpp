#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <utility> // for std::pair
#include <cassert>
#include "vk_copy.h"
#include "vk_context.h"
#include "test_class_generated.h"


std::shared_ptr<PrefSummTest> CreatePrefSummTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<PrefSummTest_Generated>();
  pObj->SetVulkanContext(a_ctx);
  pObj->InitVulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}

vk_utils::VulkanDeviceFeatures PrefSummTest_Generated_ListRequiredDeviceFeatures()
{
  vk_utils::VulkanDeviceFeatures res;
  res.features2 = PrefSummTest_Generated::ListRequiredDeviceFeatures(res.extensionNames);
  res.apiVersion = VK_API_VERSION_1_1;
  return res;
}

void PrefSummTest_Generated::InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount)
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

VkBufferUsageFlags PrefSummTest_Generated::GetAdditionalFlagsForUBO() const
{
  return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
}

uint32_t PrefSummTest_Generated::GetDefaultMaxTextures() const { return 256; }

void PrefSummTest_Generated::MakeComputePipelineAndLayout(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout* pPipelineLayout, VkPipeline* pPipeline)
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

void PrefSummTest_Generated::MakeComputePipelineOnly(const char* a_shaderPath, const char* a_mainName, const VkSpecializationInfo *a_specInfo, const VkDescriptorSetLayout a_dsLayout, VkPipelineLayout pipelineLayout, VkPipeline* pPipeline)
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

void PrefSummTest_Generated::DeleteDeviceData()
{
  if(m_commitCount == 0)
    return;
  vkDestroyBuffer(m_device, m_classDataBuffer, nullptr);
  if(copyKernelFloatDSLayout != VK_NULL_HANDLE)
     vkDestroyDescriptorSetLayout(m_device, copyKernelFloatDSLayout, nullptr);
  m_scan_int.DeleteTempBuffers(m_device);
  FreeAllAllocations(m_allMems);
}

PrefSummTest_Generated::~PrefSummTest_Generated()
{
  for(size_t i=0;i<m_allCreatedPipelines.size();i++)
    vkDestroyPipeline(m_device, m_allCreatedPipelines[i], nullptr);
  for(size_t i=0;i<m_allCreatedPipelineLayouts.size();i++)
    vkDestroyPipelineLayout(m_device, m_allCreatedPipelineLayouts[i], nullptr);
  m_scan_int.DeleteDSLayouts(m_device);
  vkDestroyDescriptorSetLayout(m_device, TestDSLayout, nullptr);
  TestDSLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorPool(m_device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;
  DeleteDeviceData();
}

void PrefSummTest_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(m_physicalDevice, &m_devProps);
}


void PrefSummTest_Generated::InitKernel_Test(const char* a_filePath)
{
  std::string shaderPath = AlterShaderPath("shaders_generated/kernel1D_Test.comp.spv");
  const VkSpecializationInfo* kspec = nullptr;
  TestDSLayout = CreateTestDSLayout();
  if(true)
  {
    MakeComputePipelineAndLayout(shaderPath.c_str(), "main", kspec, TestDSLayout, &TestLayout, &TestPipeline);
  }
  else
  {
    TestLayout   = nullptr;
    TestPipeline = nullptr;
  }
}


void PrefSummTest_Generated::InitKernels(const char* a_filePath)
{
  InitKernel_Test(a_filePath);
  // init m_scan_int
  {
    const std::string servPathFwd         = AlterShaderPath("shaders_generated/z_scan_int_block.comp.spv");
    const std::string servPathProp        = AlterShaderPath("shaders_generated/z_scan_int_propagate.comp.spv");
    m_scan_int.internalDSLayout = m_scan_int.CreateInternalScanDSLayout(m_device);
    MakeComputePipelineAndLayout(servPathFwd.c_str(),  "main", nullptr, m_scan_int.internalDSLayout, &m_scan_int.scanFwdLayout,  &m_scan_int.scanFwdPipeline);
    MakeComputePipelineAndLayout(servPathProp.c_str(), "main", nullptr, m_scan_int.internalDSLayout, &m_scan_int.scanPropLayout, &m_scan_int.scanPropPipeline);
  }
}

void PrefSummTest_Generated::InitBuffers(size_t a_maxThreadsCount, bool a_tempBuffersOverlay)
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
  {
    auto tempBuffersForScan = m_scan_int.InitTempBuffers(m_device, std::max(a_maxThreadsCount, size_t(256)));
    allBuffersRef.insert(allBuffersRef.end(), tempBuffersForScan.begin(), tempBuffersForScan.end());
  }

  auto internalBuffersMem = AllocAndBind(allBuffersRef);
  if(a_tempBuffersOverlay)
  {
    for(size_t i=0;i<groups.size();i++)
      if(i != largestIndex)
        AssignBuffersToMemory(groups[i].bufsClean, internalBuffersMem.memObject);
  }
}

void PrefSummTest_Generated::ReserveEmptyVectors()
{
}

void PrefSummTest_Generated::InitDeviceData()
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




void PrefSummTest_Generated::AssignBuffersToMemory(const std::vector<VkBuffer>& a_buffers, VkDeviceMemory a_mem)
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
      std::cout << "[PrefSummTest_Generated::AssignBuffersToMemory]: error, input buffers has different 'memReq.memoryTypeBits'" << std::endl;
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

PrefSummTest_Generated::MemLoc PrefSummTest_Generated::AllocAndBind(const std::vector<VkBuffer>& a_buffers, VkMemoryAllocateFlags a_flags)
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

PrefSummTest_Generated::MemLoc PrefSummTest_Generated::AllocAndBind(const std::vector<VkImage>& a_images, VkMemoryAllocateFlags a_flags)
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
        std::cout << "PrefSummTest_Generated::AllocAndBind(textures): memoryTypeBits warning, need to split mem allocation (override me)" << std::endl;
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

void PrefSummTest_Generated::FreeAllAllocations(std::vector<MemLoc>& a_memLoc)
{
  // in general you may check 'mem.allocId' for unique to be sure you dont free mem twice
  // for default implementation this is not needed
  for(auto mem : a_memLoc)
    vkFreeMemory(m_device, mem.memObject, nullptr);
  a_memLoc.resize(0);
}

void PrefSummTest_Generated::AllocMemoryForMemberBuffersAndImages(const std::vector<VkBuffer>& a_buffers, const std::vector<VkImage>& a_images)
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

inline size_t sblocksST(size_t elems, int threadsPerBlock)
{
  if (elems % threadsPerBlock == 0 && elems >= threadsPerBlock)
    return elems / threadsPerBlock;
  else
    return (elems / threadsPerBlock) + 1;
}

inline size_t sRoundBlocks(size_t elems, int threadsPerBlock)
{
  if (elems < threadsPerBlock)
    return (size_t)threadsPerBlock;
  else
    return sblocksST(elems, threadsPerBlock) * threadsPerBlock;
}

std::vector<VkBuffer> PrefSummTest_Generated::ScanData::InitTempBuffers(VkDevice a_device, size_t a_maxSize)
{
  m_scanMipOffsets.resize(0);
  size_t currSize = a_maxSize;
  size_t currOffset = 0;
  for (int i = 0; i < 16; i++)
  {
    size_t size2 = sRoundBlocks(currSize, 256) / 256;
    if (currSize > 0)
    {
      size_t size3 = std::max(size2, size_t(256));
      m_scanMipOffsets.push_back(currOffset);
      currOffset += size3;
    }
    else
    {
      m_scanMipOffsets.push_back(currOffset);
      currOffset += 256;
      break;
    }
    currSize = currSize / 256;
  }

  m_scanTempDataBuffer = vk_utils::createBuffer(a_device, currOffset*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_scanTempDataOffset = 0;
  m_scanMaxSize        = a_maxSize;
  return {m_scanTempDataBuffer};
}

void PrefSummTest_Generated::ScanData::DeleteTempBuffers(VkDevice a_device)
{
  vkDestroyBuffer(a_device, m_scanTempDataBuffer, nullptr);
  m_scanTempDataBuffer = VK_NULL_HANDLE;
  m_scanMipOffsets.resize(0);
}

VkDescriptorSetLayout PrefSummTest_Generated::ScanData::CreateInternalScanDSLayout(VkDevice a_device)
{
  std::array<VkDescriptorSetLayoutBinding, 3> dsBindings;

  dsBindings[0].binding            = 0;
  dsBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[0].descriptorCount    = 1;
  dsBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[0].pImmutableSamplers = nullptr;

  dsBindings[1].binding            = 1;
  dsBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[1].descriptorCount    = 1;
  dsBindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[1].pImmutableSamplers = nullptr;

  dsBindings[2].binding            = 2;
  dsBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[2].descriptorCount    = 1;
  dsBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[2].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = dsBindings.size();
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings.data();

  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

void PrefSummTest_Generated::ScanData::DeleteDSLayouts(VkDevice a_device)
{
  vkDestroyDescriptorSetLayout(a_device, internalDSLayout, nullptr);
}

void PrefSummTest_Generated::ScanData::ExclusiveScanCmd(VkCommandBuffer a_cmdBuffer, size_t a_size)
{
  InclusiveScanCmd(a_cmdBuffer, a_size, true);
}

void PrefSummTest_Generated::ScanData::InclusiveScanCmd(VkCommandBuffer a_cmdBuffer, size_t a_size, bool actuallyExclusive)
{
  if (m_scanMaxSize < a_size)
  {
    std::cout << "InclusiveScanCmd: too big input size = " << a_size << ", maximum allowed is " << m_scanMaxSize << std::endl;
    return;
  }

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  VkBufferMemoryBarrier bufBars[2] = {};
  bufBars[0].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufBars[0].pNext               = NULL;
  bufBars[0].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  bufBars[0].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
  bufBars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufBars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufBars[0].buffer              = m_scanTempDataBuffer;
  bufBars[0].offset              = 0;
  bufBars[0].size                = VK_WHOLE_SIZE;

  bufBars[1] = bufBars[0];
  bufBars[1].srcAccessMask = 0;                          // we don't going to read 'next' part of buffer in next kernel launch, just write it
  bufBars[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // we don't going to read 'next' part of buffer in next kernel launch, just write it


  size_t sizeOfElem = sizeof(uint32_t);

  uint32_t blockSizeX = 256;
  uint32_t blockSizeY = 1;
  uint32_t blockSizeZ = 1;

  struct KernelArgsPC
  {
    uint32_t iNumElementsX;
    uint32_t currMip;
    uint32_t currPassOffset;
    uint32_t nextPassOffset;
    uint32_t exclusiveFlag;
  } pcData;

  pcData.exclusiveFlag = actuallyExclusive ? 1 : 0;

  std::vector<size_t> lastSizeV;
  std::vector< std::pair<size_t,size_t> > offsets;

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scanFwdPipeline);

  // down, scan phase // fixed
  //
  int currMip = 0;
  size_t currOffset = 0;
  for (size_t currSize = a_size; currSize > 1; currSize = currSize / 256)
  {
    lastSizeV.push_back(currSize);

    const size_t runSize  = sRoundBlocks(currSize, 256);
    const size_t nextSize = runSize / 256;
    pcData.iNumElementsX  = uint32_t(runSize);
    pcData.currMip        = uint32_t(currMip);
    if(currMip == 0)
    {
      pcData.currPassOffset = 0;
      pcData.nextPassOffset = 0;
    }
    else
    {
      pcData.currPassOffset = currOffset;
      pcData.nextPassOffset = currOffset + runSize;
      offsets.push_back( std::make_pair(currOffset, currOffset + runSize) );
      currOffset += runSize;
    }

    vkCmdPushConstants(a_cmdBuffer, scanFwdLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (runSize + blockSizeX - 1) / blockSizeX, 1, 1);

    if(currMip == 0)
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    else
    {
      bufBars[0].offset = pcData.nextPassOffset*sizeOfElem;
      bufBars[0].size   = nextSize*sizeOfElem;
      bufBars[1].offset = pcData.currPassOffset*sizeOfElem;
      bufBars[1].size   = runSize*sizeOfElem;
      vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 2, bufBars, 0, nullptr);
    }

    currMip++;
  }

  currMip--;

  bufBars[0].offset = 0;
  bufBars[0].size   = VK_WHOLE_SIZE;

  vkCmdBindPipeline(a_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scanPropPipeline);

  // up, propagate phase
  //
  while (currMip >= 0)
  {
    size_t currSize = lastSizeV.back();
    lastSizeV.pop_back();

    const size_t runSize  = sRoundBlocks(currSize, 256);
    pcData.iNumElementsX  = uint32_t(runSize);
    pcData.currMip        = uint32_t(currMip);
    if(currMip == 0)
    {
      pcData.currPassOffset = 0;
      pcData.nextPassOffset = 0;
    }
    else
    {
      auto pair = offsets[currMip-1];
      pcData.currPassOffset = pair.second;
      pcData.nextPassOffset = pair.first;
    }

    vkCmdPipelineBarrier(a_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, bufBars, 0, nullptr);
    vkCmdPushConstants(a_cmdBuffer, scanFwdLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    vkCmdDispatch(a_cmdBuffer, (runSize + blockSizeX - 1) / blockSizeX, 1, 1);
    currMip--;
  }
}

VkPhysicalDeviceFeatures2 PrefSummTest_Generated::ListRequiredDeviceFeatures(std::vector<const char*>& deviceExtensions)
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

PrefSummTest_Generated::MegaKernelIsEnabled PrefSummTest_Generated::m_megaKernelFlags;

