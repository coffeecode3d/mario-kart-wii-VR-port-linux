#include <aurora/vulkan_interop.h>

#include "../internal.hpp"
#include "../stereo.hpp"
#include "gpu.hpp"

#if defined(TARGET_PC) && defined(DAWN_ENABLE_BACKEND_VULKAN) && \
    defined(MKW_AURORA_DAWN_VULKAN_NATIVE_HANDLES)

#include <dawn/native/VulkanBackend.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <unistd.h>

namespace aurora::vulkan_interop {
namespace {

Module Log("aurora::vulkan_interop");

constexpr uint32_t kOpenXREyeCount = 2;

// Vertex attribute data loaded once from the Dawn device and kept for the
// bridge lifetime. All calls are dispatched through these pointers so the TU
// never statically links libvulkan.
struct VulkanFunctions {
  // Instance-level
  PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
  PFN_vkGetPhysicalDeviceProperties getPhysicalDeviceProperties = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties = nullptr;
  // Device-level (required)
  PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;
  PFN_vkCreateImage createImage = nullptr;
  PFN_vkDestroyImage destroyImage = nullptr;
  PFN_vkGetImageMemoryRequirements getImageMemoryRequirements = nullptr;
  PFN_vkAllocateMemory allocateMemory = nullptr;
  PFN_vkFreeMemory freeMemory = nullptr;
  PFN_vkBindImageMemory bindImageMemory = nullptr;
  PFN_vkCreateCommandPool createCommandPool = nullptr;
  PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
  PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
  PFN_vkFreeCommandBuffers freeCommandBuffers = nullptr;
  PFN_vkResetCommandBuffer resetCommandBuffer = nullptr;
  PFN_vkBeginCommandBuffer beginCommandBuffer = nullptr;
  PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
  PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
  PFN_vkCmdCopyImage cmdCopyImage = nullptr;
  PFN_vkCmdCopyImageToBuffer cmdCopyImageToBuffer = nullptr;
  PFN_vkQueueSubmit queueSubmit = nullptr;
  PFN_vkQueueWaitIdle queueWaitIdle = nullptr;
  PFN_vkCreateFence createFence = nullptr;
  PFN_vkDestroyFence destroyFence = nullptr;
  PFN_vkWaitForFences waitForFences = nullptr;
  PFN_vkResetFences resetFences = nullptr;
  PFN_vkGetFenceStatus getFenceStatus = nullptr;
  PFN_vkCreateBuffer createBuffer = nullptr;
  PFN_vkDestroyBuffer destroyBuffer = nullptr;
  PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements = nullptr;
  PFN_vkBindBufferMemory bindBufferMemory = nullptr;
  PFN_vkMapMemory mapMemory = nullptr;
  PFN_vkUnmapMemory unmapMemory = nullptr;
  // VK_KHR_external_memory_fd
  PFN_vkGetMemoryFdKHR getMemoryFdKHR = nullptr;
  VkPhysicalDeviceMemoryProperties memoryProperties{};
};

VulkanFunctions g_vk{};

template <typename PFN>
static void LoadDev(VkDevice dev, const char* name, PFN& fn) {
  fn = reinterpret_cast<PFN>(g_vk.getDeviceProcAddr(dev, name));
}

template <typename PFN>
static void LoadInst(VkInstance inst, VkPhysicalDevice phys, const char* name, PFN& fn) {
  fn = reinterpret_cast<PFN>(g_vk.getInstanceProcAddr(inst, name));
}

struct NativeContextData {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;
  uint32_t queueIndex = 0;
  uint32_t apiVersion = 0;
};

static void LoadVulkanFunctions(VkInstance inst, VkPhysicalDevice phys, VkDevice dev) {
  g_vk.getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dawn::native::vulkan::GetInstanceProcAddr(
          webgpu::g_device.Get(), "vkGetInstanceProcAddr"));
  g_vk.getDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      g_vk.getInstanceProcAddr(inst, "vkGetDeviceProcAddr"));
  LoadInst(inst, phys, "vkGetPhysicalDeviceProperties", g_vk.getPhysicalDeviceProperties);
  LoadInst(inst, phys, "vkGetPhysicalDeviceMemoryProperties", g_vk.getPhysicalDeviceMemoryProperties);
  LoadInst(inst, phys, "vkGetMemoryFdKHR", g_vk.getMemoryFdKHR);
  LoadDev(dev, "vkCreateImage", g_vk.createImage);
  LoadDev(dev, "vkDestroyImage", g_vk.destroyImage);
  LoadDev(dev, "vkGetImageMemoryRequirements", g_vk.getImageMemoryRequirements);
  LoadDev(dev, "vkAllocateMemory", g_vk.allocateMemory);
  LoadDev(dev, "vkFreeMemory", g_vk.freeMemory);
  LoadDev(dev, "vkBindImageMemory", g_vk.bindImageMemory);
  LoadDev(dev, "vkCreateCommandPool", g_vk.createCommandPool);
  LoadDev(dev, "vkDestroyCommandPool", g_vk.destroyCommandPool);
  LoadDev(dev, "vkAllocateCommandBuffers", g_vk.allocateCommandBuffers);
  LoadDev(dev, "vkFreeCommandBuffers", g_vk.freeCommandBuffers);
  LoadDev(dev, "vkResetCommandBuffer", g_vk.resetCommandBuffer);
  LoadDev(dev, "vkBeginCommandBuffer", g_vk.beginCommandBuffer);
  LoadDev(dev, "vkEndCommandBuffer", g_vk.endCommandBuffer);
  LoadDev(dev, "vkCmdPipelineBarrier", g_vk.cmdPipelineBarrier);
  LoadDev(dev, "vkCmdCopyImage", g_vk.cmdCopyImage);
  LoadDev(dev, "vkCmdCopyImageToBuffer", g_vk.cmdCopyImageToBuffer);
  LoadDev(dev, "vkQueueSubmit", g_vk.queueSubmit);
  LoadDev(dev, "vkQueueWaitIdle", g_vk.queueWaitIdle);
  LoadDev(dev, "vkCreateFence", g_vk.createFence);
  LoadDev(dev, "vkDestroyFence", g_vk.destroyFence);
  LoadDev(dev, "vkWaitForFences", g_vk.waitForFences);
  LoadDev(dev, "vkResetFences", g_vk.resetFences);
  LoadDev(dev, "vkGetFenceStatus", g_vk.getFenceStatus);
  LoadDev(dev, "vkCreateBuffer", g_vk.createBuffer);
  LoadDev(dev, "vkDestroyBuffer", g_vk.destroyBuffer);
  LoadDev(dev, "vkGetBufferMemoryRequirements", g_vk.getBufferMemoryRequirements);
  LoadDev(dev, "vkBindBufferMemory", g_vk.bindBufferMemory);
  LoadDev(dev, "vkMapMemory", g_vk.mapMemory);
  LoadDev(dev, "vkUnmapMemory", g_vk.unmapMemory);

  VkPhysicalDeviceProperties props{};
  g_vk.getPhysicalDeviceProperties(phys, &props);
  g_vk.getPhysicalDeviceMemoryProperties(phys, &g_vk.memoryProperties);
}

static bool get_native_context(NativeContextData& ctx) {
  if (!webgpu::g_device || webgpu::g_backendType != wgpu::BackendType::Vulkan) {
    return false;
  }
  auto* dev = webgpu::g_device.Get();
  ctx.instance = dawn::native::vulkan::GetInstance(dev);
  ctx.physicalDevice = dawn::native::vulkan::GetVkPhysicalDevice(dev);
  ctx.device = dawn::native::vulkan::GetVkDevice(dev);
  ctx.queue = dawn::native::vulkan::GetVkQueue(dev);
  ctx.queueFamilyIndex = dawn::native::vulkan::GetGraphicsQueueFamily(dev);
  ctx.queueIndex = 0;

  VkPhysicalDeviceProperties props{};
  g_vk.getPhysicalDeviceProperties(ctx.physicalDevice, &props);
  ctx.apiVersion = props.apiVersion;
  return ctx.instance != VK_NULL_HANDLE && ctx.device != VK_NULL_HANDLE;
}

// ----------------------------------------------------------------
// Intermediate image (VkImage owned by the caller, memory exported to Dawn)
// ----------------------------------------------------------------

static constexpr int kInflight = 3;

static wgpu::TextureFormat to_wgpu_format(VkFormat fmt) noexcept {
  switch (fmt) {
  case VK_FORMAT_R8G8B8A8_UNORM:
    return wgpu::TextureFormat::RGBA8Unorm;
  case VK_FORMAT_R8G8B8A8_SRGB:
    return wgpu::TextureFormat::RGBA8UnormSrgb;
  case VK_FORMAT_B8G8R8A8_UNORM:
    return wgpu::TextureFormat::BGRA8Unorm;
  case VK_FORMAT_B8G8R8A8_SRGB:
    return wgpu::TextureFormat::BGRA8UnormSrgb;
  case VK_FORMAT_R16G16B16A16_SFLOAT:
    return wgpu::TextureFormat::RGBA16Float;
  default:
    return wgpu::TextureFormat::Undefined;
  }
}

static VkFormat to_vk_format(wgpu::TextureFormat fmt) noexcept {
  switch (fmt) {
  case wgpu::TextureFormat::RGBA8Unorm:
    return VK_FORMAT_R8G8B8A8_UNORM;
  case wgpu::TextureFormat::RGBA8UnormSrgb:
    return VK_FORMAT_R8G8B8A8_SRGB;
  case wgpu::TextureFormat::BGRA8Unorm:
    return VK_FORMAT_B8G8R8A8_UNORM;
  case wgpu::TextureFormat::BGRA8UnormSrgb:
    return VK_FORMAT_B8G8R8A8_SRGB;
  case wgpu::TextureFormat::RGBA16Float:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  default:
    return VK_FORMAT_UNDEFINED;
  }
}

struct IntermediateEye {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  wgpu::Texture texture{};
  uint32_t width = 0;
  uint32_t height = 0;
  wgpu::TextureFormat webgpuFormat = wgpu::TextureFormat::Undefined;
  VkFormat vkFormat = VK_FORMAT_UNDEFINED;
};

struct PendingTarget {
  VkImage image = VK_NULL_HANDLE;
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
};

class StereoBridge final {
public:
  StereoBridge(VkDevice device, VkQueue queue, uint32_t queueFamilyIndex,
               VkPhysicalDevice physicalDevice, uint32_t apiVersion,
               VulkanFunctions* vk, VkInstance instance,
               AuroraVulkanStereoSubmittedCallback callback, void* userdata)
      : m_device(device), m_queue(queue), m_queueFamilyIndex(queueFamilyIndex),
        m_physicalDevice(physicalDevice), m_vk(vk), m_instance(instance),
        m_callback(callback), m_userdata(userdata), m_apiVersion(apiVersion) {
    // Create command pool for the graphics queue family.
    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_queueFamilyIndex,
    };
    m_vk->createCommandPool(m_device, &poolInfo, nullptr, &m_pool);

    // One command buffer + fence per in-flight slot.
    for (uint32_t i = 0; i < kInflight; ++i) {
      const VkCommandBufferAllocateInfo allocInfo{
          .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
          .commandPool = m_pool,
          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
          .commandBufferCount = 1,
      };
      m_vk->allocateCommandBuffers(m_device, &allocInfo, &m_cmdBuffers[i]);

      const VkFenceCreateInfo fenceInfo{
          .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
          .flags = VK_FENCE_CREATE_SIGNALED_BIT,
      };
      m_vk->createFence(m_device, &fenceInfo, nullptr, &m_fences[i]);
    }
  }

  ~StereoBridge() {
    if (!m_gpuIdle) {
      (void)WaitForGpuLocked();
    }
    // Destroy intermediates.
    for (auto& intermediate : m_intermediates) {
      DestroyIntermediate(intermediate);
    }
    // Destroy command buffers + fences + pool.
    for (uint32_t i = 0; i < kInflight; ++i) {
      if (m_cmdBuffers[i] != VK_NULL_HANDLE) {
        m_vk->freeCommandBuffers(m_device, m_pool, 1, &m_cmdBuffers[i]);
      }
      if (m_fences[i] != VK_NULL_HANDLE) {
        m_vk->destroyFence(m_device, m_fences[i], nullptr);
      }
    }
    if (m_pool != VK_NULL_HANDLE) {
      m_vk->destroyCommandPool(m_device, m_pool, nullptr);
    }
  }

  bool PrepareForDestruction() noexcept {
    std::lock_guard lock(m_mutex);
    return WaitForGpuLocked();
  }

  bool SetTargets(uint64_t token, const AuroraVulkanStereoTarget* targets,
                  uint32_t targetCount) noexcept {
    if (token == 0 || targets == nullptr || targetCount == 0 ||
        targetCount > kOpenXREyeCount) {
      return false;
    }
    std::lock_guard lock(m_mutex);
    if (m_framePending || m_encoded) {
      return false;
    }
    for (uint32_t eye = 0; eye < targetCount; ++eye) {
      if (targets[eye].image == 0 || targets[eye].width == 0 ||
          targets[eye].height == 0 ||
          targets[eye].format == VK_FORMAT_UNDEFINED) {
        return false;
      }
      m_targets[eye] = {
          .image = reinterpret_cast<VkImage>(targets[eye].image),
          .width = targets[eye].width,
          .height = targets[eye].height,
          .format = static_cast<VkFormat>(targets[eye].format),
      };
    }
    for (uint32_t eye = targetCount; eye < m_targets.size(); ++eye) {
      m_targets[eye] = {};
    }
    m_frameToken = token;
    m_targetCount = targetCount;
    m_framePending = true;
    return true;
  }

  bool Encode(wgpu::CommandEncoder& encoder, const stereo::SinkFrame& frame) noexcept {
    std::lock_guard lock(m_mutex);
    if (!m_framePending || m_encoded || frame.frameToken != m_frameToken) {
      return false;
    }
    if (EncodeLocked(encoder, frame)) {
      m_encoded = true;
      return true;
    }
    PublishAndClearFrameLocked(frame.frameToken, false);
    return false;
  }

  void Submitted(const stereo::SinkFrame& frame) noexcept {
    std::lock_guard lock(m_mutex);
    if (!m_framePending || !m_encoded || frame.frameToken != m_frameToken) {
      return;
    }
    const bool success = EnqueueNativeCopyLocked();
    CompleteCoupledCapture();
    PublishAndClearFrameLocked(frame.frameToken, success);
  }

  void CancelPending() noexcept {
    std::lock_guard lock(m_mutex);
    if (!m_framePending) {
      return;
    }
    PublishAndClearFrameLocked(m_frameToken, false);
  }

  bool CancelBeforeEncode(uint64_t token) noexcept {
    std::unique_lock lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
      return false;
    }
    if (token == 0 || !m_framePending || m_encoded || token != m_frameToken) {
      return false;
    }
    ClearFrameLocked();
    return true;
  }

private:
  bool sameCopyFamily(VkFormat a, VkFormat b) noexcept {
    auto family = [](VkFormat f) {
      switch (f) {
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_R8G8B8A8_SRGB:
        return 1;
      case VK_FORMAT_B8G8R8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_SRGB:
        return 2;
      case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 3;
      default:
        return 0;
      }
    };
    int fa = family(a);
    return fa != 0 && fa == family(b);
  }

  // Select a device-local, exportable (OPAQUE_FD) memory type index
  // for the given VkImage memory requirements.
  bool SelectExportableMemoryType(VkMemoryRequirements req, uint32_t& outTypeIndex,
                                  VkDeviceSize& outAllocationSize) noexcept {
    // Prefer device-local types. Try those first, then fall back to any type
    // that accepts export. An allocation with VkExportMemoryAllocateInfo is
    // necessary so that vkGetMemoryFdKHR succeeds.
    const VkExportMemoryAllocateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    for (uint32_t retry = 0; retry < 2; ++retry) {
      const VkMemoryPropertyFlags requiredProps =
          (retry == 0) ? (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) : 0;
      for (uint32_t i = 0; i < m_vk->memoryProperties.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) == 0) {
          continue;
        }
        const VkMemoryPropertyFlags props =
            m_vk->memoryProperties.memoryTypes[i].propertyFlags;
        if (requiredProps != 0 && (props & requiredProps) == 0) {
          continue;
        }
        // Try a small alloc to verify export is allowed.
        const VkMemoryDedicatedAllocateInfo dedicatedInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        };
        const VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &exportInfo,
            .allocationSize = req.size,
            .memoryTypeIndex = i,
        };
        VkDeviceMemory testAlloc = VK_NULL_HANDLE;
        if (m_vk->allocateMemory(m_device, &allocInfo, nullptr, &testAlloc) !=
            VK_SUCCESS) {
          continue;
        }
        VkMemoryGetFdInfoKHR getFdInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            .memory = testAlloc,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        int fd = -1;
        VkResult result = m_vk->getMemoryFdKHR(m_device, &getFdInfo, &fd);
        if (result == VK_SUCCESS && fd >= 0) {
          close(fd);
          m_vk->freeMemory(m_device, testAlloc, nullptr);
          outTypeIndex = i;
          outAllocationSize = req.size;
          return true;
        }
        m_vk->freeMemory(m_device, testAlloc, nullptr);
      }
    }
    return false;
  }

  bool EnsureIntermediate(uint32_t eye, const stereo::EyeImage& source) noexcept {
    auto& intermediate = m_intermediates[eye];
    const VkFormat sourceFormat = to_vk_format(source.format);
    if (source.texture == nullptr || sourceFormat == VK_FORMAT_UNDEFINED ||
        source.size.width != m_targets[eye].width ||
        source.size.height != m_targets[eye].height ||
        !sameCopyFamily(sourceFormat, m_targets[eye].format)) {
      Log.error("Stereo eye {} does not match its OpenXR Vulkan target", eye);
      return false;
    }
    if (intermediate.image != VK_NULL_HANDLE && intermediate.width == source.size.width &&
        intermediate.height == source.size.height && intermediate.webgpuFormat == source.format) {
      return true;
    }

    DestroyIntermediate(intermediate);

    // Create native VkImage + exportable memory + import into Dawn.
    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = sourceFormat,
        .extent = {source.size.width, source.size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        // Dawn's external import (MemoryServiceImplementationOpaqueFD.cpp)
        // re-creates the image with |VK_IMAGE_CREATE_ALIAS_BIT_KHR| forced into
        // the flags when it wraps the exported memory. The native intermediate
        // that Phase B reads back from must use the SAME createInfo flags (and
        // therefore the same driver-chosen tiling/swizzle), or the pixels Dawn
        // wrote through its re-created image are decoded by the native image in
        // a scrambled, tile-periodic order.
        .flags = VK_IMAGE_CREATE_ALIAS_BIT_KHR,
    };
    if (m_vk->createImage(m_device, &imageInfo, nullptr, &intermediate.image) !=
        VK_SUCCESS) {
      Log.error("Could not create Vulkan stereo intermediate for eye {}", eye);
      return false;
    }

    VkMemoryRequirements req{};
    m_vk->getImageMemoryRequirements(m_device, intermediate.image, &req);

    uint32_t typeIndex = 0;
    VkDeviceSize allocationSize = 0;
    if (!SelectExportableMemoryType(req, typeIndex, allocationSize)) {
      Log.error("Could not find exportable memory type for stereo intermediate eye {}", eye);
      DestroyIntermediate(intermediate);
      return false;
    }

    // The exported memory must use a DEDICATED allocation on the native side,
    // mirroring how Dawn binds the imported memory: Dawn queries
    // VkMemoryDedicatedRequirements and, when the image prefers a dedicated
    // allocation (the norm on these drivers), re-imports the FD as a dedicated
    // allocation owned by its re-created image. A plain, non-dedicated binding
    // makes the driver select a DIFFERENT tile/swizzle layout for our native
    // intermediate than for Dawn's image over the SAME memory, which surfaces
    // as the tile-periodic (16-texel vertical stripes) scramble in the headset.
    // Binding our own native image as a dedicated allocation forces both sides
    // onto the same swizzled layout.
    const VkMemoryDedicatedAllocateInfo dedicatedInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .image = intermediate.image,
        .buffer = VK_NULL_HANDLE,
    };
    const VkExportMemoryAllocateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicatedInfo,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    const VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &exportInfo,
        .allocationSize = allocationSize,
        .memoryTypeIndex = typeIndex,
    };
    if (m_vk->allocateMemory(m_device, &allocInfo, nullptr, &intermediate.memory) !=
        VK_SUCCESS) {
      Log.error("Could not allocate exportable memory for stereo intermediate eye {}", eye);
      DestroyIntermediate(intermediate);
      return false;
    }
    if (m_vk->bindImageMemory(m_device, intermediate.image, intermediate.memory, 0) !=
        VK_SUCCESS) {
      Log.error("Could not bind memory for stereo intermediate eye {}", eye);
      DestroyIntermediate(intermediate);
      return false;
    }

    // Export the memory to a file descriptor and import into Dawn.
    VkMemoryGetFdInfoKHR getFdInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = intermediate.memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    int exportedFd = -1;
    if (m_vk->getMemoryFdKHR(m_device, &getFdInfo, &exportedFd) != VK_SUCCESS ||
        exportedFd < 0) {
      Log.error("Could not export FD for stereo intermediate eye {}", eye);
      DestroyIntermediate(intermediate);
      return false;
    }

    const WGPUTextureDescriptor textureDescriptor{
        .label = eye == 0 ? "Aurora Vulkan stereo L intermediate"
                          : "Aurora Vulkan stereo R intermediate",
        // Both transfer directions must be present so the VkImage Dawn
        // re-creates over the exported memory uses the SAME usage bits (and
        // therefore the same block-linear tiling/swizzle) as the native
        // intermediate that Phase B copies out of. With CopyDst alone, Dawn
        // produced a transfer-only image whose layout disagreed on some
        // drivers, which surfaced as tile-periodic (16-texel) vertical stripes
        // in the headset.
        .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc,
        .dimension = WGPUTextureDimension_2D,
        .size = {source.size.width, source.size.height, 1},
        .format = static_cast<WGPUTextureFormat>(source.format),
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    dawn::native::vulkan::ExternalImageDescriptorOpaqueFD fdDescriptor{};
    fdDescriptor.memoryFD = exportedFd;
    fdDescriptor.allocationSize = allocationSize;
    fdDescriptor.memoryTypeIndex = typeIndex;
    fdDescriptor.releasedOldLayout = VK_IMAGE_LAYOUT_GENERAL;
    fdDescriptor.releasedNewLayout = VK_IMAGE_LAYOUT_GENERAL;
    fdDescriptor.cTextureDescriptor = &textureDescriptor;

    WGPUTexture rawTexture =
        dawn::native::vulkan::WrapVulkanImage(webgpu::g_device.Get(), &fdDescriptor);
    // Dawn takes ownership of the FD on successful import. On failure the FD
    // is still consumed (ownership transferred), so close unconditionally.
    close(exportedFd);
    if (rawTexture == nullptr) {
      Log.error("Dawn rejected Vulkan stereo intermediate for eye {}", eye);
      DestroyIntermediate(intermediate);
      return false;
    }
    intermediate.texture = wgpu::Texture::Acquire(rawTexture);
    intermediate.width = source.size.width;
    intermediate.height = source.size.height;
    intermediate.webgpuFormat = source.format;
    intermediate.vkFormat = sourceFormat;
    return true;
  }

  bool EncodeLocked(wgpu::CommandEncoder& encoder, const stereo::SinkFrame& frame) noexcept {
    for (uint32_t eye = 0; eye < m_targetCount; ++eye) {
      if (!EnsureIntermediate(eye, frame.eyes[eye])) {
        return false;
      }
    }
    ArmCoupledCapture(frame);
    // Phase A: Dawn copies the aurora-rendered eye texture into our native
    // intermediate (wrapped into a Dawn wgpu::Texture via OpaqueFD import).
    for (uint32_t eye = 0; eye < m_targetCount; ++eye) {
      const wgpu::TexelCopyTextureInfo source{
          .texture = *frame.eyes[eye].texture,
          .mipLevel = 0,
          .origin = {},
          .aspect = wgpu::TextureAspect::All,
      };
      const wgpu::TexelCopyTextureInfo destination{
          .texture = m_intermediates[eye].texture,
          .mipLevel = 0,
          .origin = {},
          .aspect = wgpu::TextureAspect::All,
      };
      const wgpu::Extent3D extent{m_intermediates[eye].width, m_intermediates[eye].height, 1};
      encoder.CopyTextureToTexture(&source, &destination, &extent);
    }
    EncodeCoupledCapture(encoder, frame);
    return true;
  }

  // Phase B: native vkCmdCopyImage from our intermediate (same device memory
  // aliased by Dawn's wrapped texture) into the acquired XR swapchain image.
  bool EnqueueNativeCopyLocked() noexcept {
    CollectCompletedCommandsLocked();

    // Slot 0 is sufficient for single-frame-per-swapchain, but we round-robin
    // to keep fence management simple.
    const uint32_t slot = m_nextSlot % kInflight;
    m_nextSlot = (m_nextSlot + 1) % kInflight;
    if (m_vk->waitForFences(m_device, 1, &m_fences[slot], VK_TRUE,
                            UINT64_MAX) != VK_SUCCESS) {
      Log.error("Timeout waiting for Vulkan stereo fence slot {}", slot);
      return false;
    }
    m_vk->resetFences(m_device, 1, &m_fences[slot]);

    VkCommandBuffer cmd = m_cmdBuffers[slot];
    m_vk->resetCommandBuffer(cmd, 0);

    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (m_vk->beginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
      Log.error("Could not begin Vulkan stereo command buffer");
      return false;
    }

    for (uint32_t eye = 0; eye < m_targetCount; ++eye) {
      const VkImage sourceImage = m_intermediates[eye].image;
      const VkImage dstImage = m_targets[eye].image;
      const uint32_t w = m_intermediates[eye].width;
      const uint32_t h = m_intermediates[eye].height;

      // Make Dawn's writes visible.
      const VkImageMemoryBarrier barrierSrc{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
          .newLayout = VK_IMAGE_LAYOUT_GENERAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = sourceImage,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      m_vk->cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                               0, nullptr, 1, &barrierSrc);

      // Transition XR target to TRANSFER_DST.
      const VkImageMemoryBarrier barrierDstBefore{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = 0,
          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
          .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = dstImage,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      m_vk->cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                               0, nullptr, 1, &barrierDstBefore);

      const VkImageCopy region{
          .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .extent = {w, h, 1},
      };
      m_vk->cmdCopyImage(cmd, sourceImage, VK_IMAGE_LAYOUT_GENERAL,
                         dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         1, &region);

      // Restore XR target to GENERAL for the compositor / next acquisition.
      const VkImageMemoryBarrier barrierDstAfter{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = 0,
          .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          .newLayout = VK_IMAGE_LAYOUT_GENERAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = dstImage,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      m_vk->cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                               nullptr, 0, nullptr, 1, &barrierDstAfter);
    }

    if (m_vk->endCommandBuffer(cmd) != VK_SUCCESS) {
      Log.error("Could not end Vulkan stereo command buffer");
      return false;
    }

    // Submit to the SAME queue Dawn is using. The frame worker has already
    // submitted its own work to this queue, so submission order guarantees
    // our native copy sees the Dawn-encoded pixels.
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    if (m_vk->queueSubmit(m_queue, 1, &submitInfo, m_fences[slot]) != VK_SUCCESS) {
      Log.error("Could not submit Vulkan stereo command buffer");
      return false;
    }
    m_gpuIdle = false;
    return true;
  }

  void CollectCompletedCommandsLocked() {
    // All fence slots are checked when the frame is released; no list
    // management required (kInflight is small and fences are waited eagerly).
  }

  void DestroyIntermediate(IntermediateEye& i) {
    if (i.texture) {
      i.texture = {};
    }
    if (i.image != VK_NULL_HANDLE) {
      m_vk->destroyImage(m_device, i.image, nullptr);
      i.image = VK_NULL_HANDLE;
    }
    if (i.memory != VK_NULL_HANDLE) {
      m_vk->freeMemory(m_device, i.memory, nullptr);
      i.memory = VK_NULL_HANDLE;
    }
    i.width = 0;
    i.height = 0;
    i.webgpuFormat = wgpu::TextureFormat::Undefined;
    i.vkFormat = VK_FORMAT_UNDEFINED;
  }

  void ClearFrameLocked() noexcept {
    for (auto& target : m_targets) {
      target = {};
    }
    m_frameToken = 0;
    m_targetCount = 0;
    m_framePending = false;
    m_encoded = false;
  }

  // Diagnostic (MKW_VR_CAPTURE_INTERMEDIATE=1): on the first submitted stereo
  // frame, read back BOTH the aurora eye texture and the Dawn-wrapped
  // intermediate on the SAME frame so the two can be compared bit-for-bit.
  // This separates "Dawn wrote garbage into shared memory" (Phase A) from
  // "native copy corrupts a good intermediate" (Phase B). The copies are
  // appended to the sink encoder, so no command is queued out of order.
  bool ArmCoupledCapture(const stereo::SinkFrame& frame) noexcept {
    static bool armed = std::getenv("MKW_VR_CAPTURE_INTERMEDIATE") != nullptr;
    if (!armed) {
      return false;
    }
    const uint32_t w = frame.eyes[0].size.width;
    const uint32_t h = frame.eyes[0].size.height;
    if (w == 0 || h == 0 || frame.eyes[0].texture == nullptr ||
        m_intermediates[0].texture == nullptr) {
      return false;
    }
    armed = false;
    const uint32_t bytesPerRow = ((w * 4 + 255) / 256) * 256;
    const uint64_t size = static_cast<uint64_t>(bytesPerRow) * h;
    const wgpu::BufferDescriptor desc{
        .label = "Coupled capture staging",
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        .size = size,
    };
    m_captureBuffers[0] = webgpu::g_device.CreateBuffer(&desc);
    m_captureBuffers[1] = webgpu::g_device.CreateBuffer(&desc);
    m_captureBytesPerRow = bytesPerRow;
    m_captureWidth = w;
    m_captureHeight = h;
    m_captureArmed = true;
    return true;
  }

  void EncodeCoupledCapture(wgpu::CommandEncoder& encoder, const stereo::SinkFrame& frame) noexcept {
    if (!m_captureArmed) {
      return;
    }
    auto copy = [&](wgpu::Texture texture, wgpu::Buffer buffer) {
      const wgpu::TexelCopyTextureInfo src{
          .texture = texture,
          .mipLevel = 0,
          .origin = {},
          .aspect = wgpu::TextureAspect::All,
      };
      const wgpu::TexelCopyBufferInfo dst{
          .layout = {.offset = 0, .bytesPerRow = m_captureBytesPerRow, .rowsPerImage = m_captureHeight},
          .buffer = buffer,
      };
      const wgpu::Extent3D extent{m_captureWidth, m_captureHeight, 1};
      encoder.CopyTextureToBuffer(&src, &dst, &extent);
    };
    copy(*frame.eyes[0].texture, m_captureBuffers[0]);
    copy(m_intermediates[0].texture, m_captureBuffers[1]);
    m_captureArmed = false;
    m_captureQueued = true;
  }

  void CompleteCoupledCapture() noexcept {
    if (!m_captureQueued) {
      return;
    }
    m_captureQueued = false;
    wgpu::MapAsyncStatus status1 = wgpu::MapAsyncStatus::CallbackCancelled;
    wgpu::MapAsyncStatus status2 = wgpu::MapAsyncStatus::CallbackCancelled;
    wgpu::StringView msg1{};
    wgpu::StringView msg2{};
    const auto f1 = m_captureBuffers[0].MapAsync(
        wgpu::MapMode::Read, 0, m_captureBufferSize(), wgpu::CallbackMode::WaitAnyOnly,
        [&status1, &msg1](wgpu::MapAsyncStatus status, wgpu::StringView m) {
          status1 = status;
          msg1 = m;
        });
    const auto f2 = m_captureBuffers[1].MapAsync(
        wgpu::MapMode::Read, 0, m_captureBufferSize(), wgpu::CallbackMode::WaitAnyOnly,
        [&status2, &msg2](wgpu::MapAsyncStatus status, wgpu::StringView m) {
          status2 = status;
          msg2 = m;
        });
    const auto w1 = webgpu::g_instance.WaitAny(f1, 8000000000);
    const auto w2 = webgpu::g_instance.WaitAny(f2, 8000000000);
    Log.info("Coupled capture map: wait1={} status1={} msg1={} wait2={} status2={} msg2={}",
             static_cast<int>(w1), static_cast<int>(status1), msg1,
             static_cast<int>(w2), static_cast<int>(status2), msg2);
    {
      std::ofstream out("coupled_capture_status.txt", std::ios::out | std::ios::trunc);
      if (out) {
        const std::string m1 = msg1.data && msg1.length != 0 ? std::string(msg1.data, msg1.length) : "";
        const std::string m2 = msg2.data && msg2.length != 0 ? std::string(msg2.data, msg2.length) : "";
        out << "wait1=" << static_cast<int>(w1) << " status1=" << static_cast<int>(status1)
            << " msg1=" << m1 << " wait2=" << static_cast<int>(w2)
            << " status2=" << static_cast<int>(status2) << " msg2=" << m2 << "\n";
      }
    }
    const uint8_t* eye =
        status1 == wgpu::MapAsyncStatus::Success
            ? static_cast<const uint8_t*>(m_captureBuffers[0].GetConstMappedRange(0, m_captureBufferSize()))
            : nullptr;
    const uint8_t* intermediate =
        status2 == wgpu::MapAsyncStatus::Success
            ? static_cast<const uint8_t*>(m_captureBuffers[1].GetConstMappedRange(0, m_captureBufferSize()))
            : nullptr;
    if (eye) {
      WriteBmp("coupled_eye_left.bmp", eye, m_captureWidth, m_captureHeight, m_captureBytesPerRow);
    }
    if (intermediate) {
      WriteBmp("coupled_intermediate_left.bmp", intermediate, m_captureWidth, m_captureHeight, m_captureBytesPerRow);
    }
    if (m_captureBuffers[0]) {
      m_captureBuffers[0].Unmap();
    }
    if (m_captureBuffers[1]) {
      m_captureBuffers[1].Unmap();
    }
    m_captureBuffers[0] = {};
    m_captureBuffers[1] = {};
    CaptureNativePerspectiveLeftBmp();
    Log.info("Wrote coupled eye/intermediate capture");
  }

  // Phase A happened on the WebGPU queue before our native copy slot fence was
  // signalled; wait for the queue to drain so the native image definitely has
  // Dawn's pixels, then read the SAME shared memory through the NATIVE image's
  // layout (the view Phase B blits into the swapchain). Compare this BMP to
  // coupled_intermediate_left.bmp: if they differ, Dawn's wrapped image and the
  // native intermediate use different tilings of the same memory.
  void CaptureNativePerspectiveLeftBmp() noexcept {
    auto& intermediate = m_intermediates[0];
    if (intermediate.image == VK_NULL_HANDLE || intermediate.width == 0) {
      return;
    }
    m_vk->queueWaitIdle(m_queue);
    uint32_t slot = 0;
    for (uint32_t i = 0; i < kInflight; ++i) {
      if (i != m_nextSlot) {
        slot = i;
        break;
      }
    }
    if (m_vk->waitForFences(m_device, 1, &m_fences[slot], VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
      return;
    }
    const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(intermediate.width) *
                                    static_cast<VkDeviceSize>(intermediate.height) * 4;
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bufferSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    if (m_vk->createBuffer(m_device, &bufferInfo, nullptr, &staging) != VK_SUCCESS) {
      return;
    }
    VkMemoryRequirements memReq{};
    m_vk->getBufferMemoryRequirements(m_device, staging, &memReq);
    uint32_t memIndex = 0;
    bool found = false;
    for (uint32_t i = 0; i < m_vk->memoryProperties.memoryTypeCount; ++i) {
      if ((memReq.memoryTypeBits & (1u << i)) != 0 &&
          (m_vk->memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        memIndex = i;
        found = true;
        break;
      }
    }
    if (!found) {
      m_vk->destroyBuffer(m_device, staging, nullptr);
      return;
    }
    const VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReq.size,
        .memoryTypeIndex = memIndex,
    };
    if (m_vk->allocateMemory(m_device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
      m_vk->destroyBuffer(m_device, staging, nullptr);
      return;
    }
    m_vk->bindBufferMemory(m_device, staging, stagingMemory, 0);
    VkCommandBuffer cmd = m_cmdBuffers[slot];
    m_vk->resetCommandBuffer(cmd, 0);
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    m_vk->beginCommandBuffer(cmd, &beginInfo);
    const VkImageMemoryBarrier barrierSrc{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = intermediate.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    m_vk->cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &barrierSrc);
    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {intermediate.width, intermediate.height, 1},
    };
    m_vk->cmdCopyImageToBuffer(cmd, intermediate.image, VK_IMAGE_LAYOUT_GENERAL, staging, 1, &region);
    m_vk->endCommandBuffer(cmd);
    m_vk->resetFences(m_device, 1, &m_fences[slot]);
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    if (m_vk->queueSubmit(m_queue, 1, &submitInfo, m_fences[slot]) != VK_SUCCESS ||
        m_vk->waitForFences(m_device, 1, &m_fences[slot], VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
      m_vk->freeMemory(m_device, stagingMemory, nullptr);
      m_vk->destroyBuffer(m_device, staging, nullptr);
      return;
    }
    void* mapped = nullptr;
    if (m_vk->mapMemory(m_device, stagingMemory, 0, bufferSize, 0, &mapped) == VK_SUCCESS) {
      WriteBmp("coupled_intermediate_native_left.bmp", static_cast<const uint8_t*>(mapped),
               intermediate.width, intermediate.height, intermediate.width * 4);
      m_vk->unmapMemory(m_device, stagingMemory);
    }
    m_vk->freeMemory(m_device, stagingMemory, nullptr);
    m_vk->destroyBuffer(m_device, staging, nullptr);
  }

  VkDeviceSize m_captureBufferSize() const noexcept {
    return static_cast<VkDeviceSize>(m_captureBytesPerRow) * m_captureHeight;
  }

  static void WriteBmp(const char* path, const uint8_t* rgba, uint32_t width,
                       uint32_t height, uint32_t bytesPerRow) noexcept {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      return;
    }
    constexpr uint32_t pixelOffset = 14 + 40;
    const uint32_t imageSize = width * height * 4;
    auto writeU32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto writeU16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    out.write("BM", 2);
    writeU32(pixelOffset + imageSize);
    writeU16(0);
    writeU16(0);
    writeU32(pixelOffset);
    writeU32(40);
    writeU32(width);
    writeU32(height);
    writeU16(1);
    writeU16(32);
    writeU32(0);
    writeU32(imageSize);
    writeU32(2835);
    writeU32(2835);
    writeU32(0);
    writeU32(0);
    std::vector<uint8_t> row(static_cast<size_t>(width) * 4);
    for (uint32_t y = height; y-- > 0;) {
      const uint8_t* source = rgba + static_cast<size_t>(y) * bytesPerRow;
      for (uint32_t x = 0; x < width; ++x) {
        row[x * 4 + 0] = source[x * 4 + 2];
        row[x * 4 + 1] = source[x * 4 + 1];
        row[x * 4 + 2] = source[x * 4 + 0];
        row[x * 4 + 3] = source[x * 4 + 3];
      }
      out.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
  }

  void PublishAndClearFrameLocked(uint64_t token, bool success) noexcept {
    Notify(token, success);
    ClearFrameLocked();
  }

  void Notify(uint64_t token, bool success) noexcept {
    if (m_callback != nullptr) {
      m_callback(token, success, m_userdata);
    }
  }

  bool WaitForGpuLocked() noexcept {
    if (m_gpuIdle) {
      return true;
    }
    if (m_vk->queueWaitIdle(m_queue) != VK_SUCCESS) {
      Log.error("Timeout waiting for Vulkan queue to become idle");
      return false;
    }
    // Reset fences so they are signalled (ready for next waitForFences).
    for (uint32_t i = 0; i < kInflight; ++i) {
      m_vk->resetFences(m_device, 1, &m_fences[i]);
    }
    m_gpuIdle = true;
    return true;
  }

  std::mutex m_mutex;
  VkDevice m_device = VK_NULL_HANDLE;
  VkQueue m_queue = VK_NULL_HANDLE;
  uint32_t m_queueFamilyIndex = 0;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VulkanFunctions* m_vk = nullptr;
  VkInstance m_instance = VK_NULL_HANDLE;
  uint32_t m_apiVersion = 0;
  VkCommandPool m_pool = VK_NULL_HANDLE;
  std::array<VkCommandBuffer, kInflight> m_cmdBuffers{};
  std::array<VkFence, kInflight> m_fences{};
  uint32_t m_nextSlot = 0;
  std::array<IntermediateEye, kOpenXREyeCount> m_intermediates{};
  std::array<PendingTarget, kOpenXREyeCount> m_targets{};
  AuroraVulkanStereoSubmittedCallback m_callback = nullptr;
  void* m_userdata = nullptr;
  uint64_t m_frameToken = 0;
  uint32_t m_targetCount = 0;
  bool m_framePending = false;
  bool m_encoded = false;
  bool m_gpuIdle = true;
  std::array<wgpu::Buffer, 2> m_captureBuffers{};
  uint32_t m_captureBytesPerRow = 0;
  uint32_t m_captureWidth = 0;
  uint32_t m_captureHeight = 0;
  bool m_captureArmed = false;
  bool m_captureQueued = false;
};

std::unique_ptr<StereoBridge> g_bridge;
VkDevice g_device = VK_NULL_HANDLE;
std::once_flag g_vkFunctionsOnce;

bool loadFunctionsOnce() {
  if (!webgpu::g_device || webgpu::g_backendType != wgpu::BackendType::Vulkan) {
    return false;
  }
  auto* dev = webgpu::g_device.Get();
  VkInstance inst = dawn::native::vulkan::GetInstance(dev);
  VkPhysicalDevice phys = dawn::native::vulkan::GetVkPhysicalDevice(dev);
  VkDevice vkDev = dawn::native::vulkan::GetVkDevice(dev);
  std::call_once(g_vkFunctionsOnce, [inst, phys, vkDev]() {
    LoadVulkanFunctions(inst, phys, vkDev);
    g_device = vkDev;
  });
  return g_device != VK_NULL_HANDLE;
}

bool sink_encode(wgpu::CommandEncoder& encoder, const stereo::SinkFrame& frame,
                 void* userdata) noexcept {
  return static_cast<StereoBridge*>(userdata)->Encode(encoder, frame);
}

void sink_submitted(const stereo::SinkFrame& frame, void* userdata) noexcept {
  static_cast<StereoBridge*>(userdata)->Submitted(frame);
}

} // namespace
} // namespace aurora::vulkan_interop

extern "C" {

bool aurora_vulkan_get_native_handles(AuroraVulkanNativeContext* context) {
  if (context == nullptr) {
    return false;
  }
  *context = {};
  using namespace aurora::vulkan_interop;
  if (!loadFunctionsOnce()) {
    return false;
  }
  NativeContextData data{};
  if (!get_native_context(data)) {
    return false;
  }
  const int64_t colorFormat =
      to_vk_format(aurora::webgpu::g_graphicsConfig.surfaceConfiguration.format);
  context->version = 1;
  context->flags =
      AURORA_VULKAN_NATIVE_CONTEXT_DAWN_OWNED | AURORA_VULKAN_NATIVE_CONTEXT_QUEUE_VERIFIED;
  context->instance = reinterpret_cast<uint64_t>(data.instance);
  context->physicalDevice = reinterpret_cast<uint64_t>(data.physicalDevice);
  context->device = reinterpret_cast<uint64_t>(data.device);
  context->queue = reinterpret_cast<uint64_t>(data.queue);
  context->queueFamilyIndex = data.queueFamilyIndex;
  context->queueIndex = data.queueIndex;
  context->apiVersion = data.apiVersion;
  context->colorFormat = static_cast<uint32_t>(colorFormat);
  context->targetCountLimit = AURORA_VULKAN_STEREO_MAX_TARGETS;
  context->acquireQueueLock = [](void*) -> bool {
    aurora::renderer_gpu_mutex().lock();
    return true;
  };
  context->releaseQueueLock = [](void*) { aurora::renderer_gpu_mutex().unlock(); };
  return true;
}

bool aurora_vulkan_enable_stereo_bridge(AuroraVulkanStereoSubmittedCallback submitted,
                                        void* userdata) {
  using namespace aurora::vulkan_interop;
  if (g_bridge || submitted == nullptr) {
    return false;
  }
  if (!loadFunctionsOnce()) {
    return false;
  }
  NativeContextData data{};
  if (!get_native_context(data)) {
    return false;
  }
  auto bridge = std::make_unique<StereoBridge>(
      data.device, data.queue, data.queueFamilyIndex, data.physicalDevice,
      data.apiVersion, &g_vk,
      data.instance, submitted, userdata);
  aurora::stereo::set_sink(sink_encode, sink_submitted, bridge.get());
  g_bridge = std::move(bridge);
  return true;
}

bool aurora_vulkan_set_stereo_targets(uint64_t frameToken,
                                      const AuroraVulkanStereoTarget* targets,
                                      uint32_t targetCount) {
  using namespace aurora::vulkan_interop;
  return g_bridge && g_bridge->SetTargets(frameToken, targets, targetCount);
}

bool aurora_vulkan_cancel_stereo_targets(uint64_t frameToken) {
  using namespace aurora::vulkan_interop;
  return g_bridge && g_bridge->CancelBeforeEncode(frameToken);
}

bool aurora_vulkan_disable_stereo_bridge() {
  using namespace aurora::vulkan_interop;
  if (!g_bridge) {
    return true;
  }
  aurora::stereo::set_sink(nullptr, nullptr, nullptr);
  g_bridge->CancelPending();
  if (!g_bridge->PrepareForDestruction()) {
    (void)g_bridge.release();
    return false;
  }
  g_bridge.reset();
  return true;
}

} // extern "C"

#else

bool aurora_vulkan_get_native_handles(AuroraVulkanNativeContext* context) {
  if (context != nullptr) {
    *context = {};
  }
  return false;
}

bool aurora_vulkan_enable_stereo_bridge(AuroraVulkanStereoSubmittedCallback, void*) {
  return false;
}

bool aurora_vulkan_set_stereo_targets(uint64_t,
                                      const AuroraVulkanStereoTarget*,
                                      uint32_t) {
  return false;
}

bool aurora_vulkan_cancel_stereo_targets(uint64_t) { return false; }

bool aurora_vulkan_disable_stereo_bridge() { return true; }

#endif