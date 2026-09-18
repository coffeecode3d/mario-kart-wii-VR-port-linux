#ifndef AURORA_VULKAN_INTEROP_H
#define AURORA_VULKAN_INTEROP_H

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include "stdbool.h"
#include "stdint.h"
#endif

enum { AURORA_VULKAN_STEREO_MAX_TARGETS = 2 };

/**
 * Borrowed native objects owned by Aurora/Dawn. They remain valid until
 * aurora_shutdown() and must not be released by the caller.
 *
 * colorFormat is the VkFormat matching Aurora's single-sample eye output.
 * apiVersion is encoded with VK_MAKE_API_VERSION(0, major, minor, patch).
 *
 * The lock callbacks pair with Aurora's own graphics-queue submission mutex.
 * They must be held around any Vulkan work (command recording, queue submits,
 * waits) that is not already synchronized by Aurora's frame worker, in the
 * same way the D3D12 interop relies on that backend's queue lock.
 */
typedef struct {
  uint32_t version;
  uint32_t flags; /* AURORA_VULKAN_NATIVE_CONTEXT_DAWN_OWNED |
                     AURORA_VULKAN_NATIVE_CONTEXT_QUEUE_VERIFIED */
  uint64_t instance;
  uint64_t physicalDevice;
  uint64_t device;
  uint64_t queue;
  uint32_t queueFamilyIndex;
  uint32_t queueIndex;
  uint32_t apiVersion;
  uint32_t colorFormat;
  uint32_t targetCountLimit;
  bool (*acquireQueueLock)(void* userdata);
  void (*releaseQueueLock)(void* userdata);
  void* queueLockUserdata;
} AuroraVulkanNativeContext;

enum {
  AURORA_VULKAN_NATIVE_CONTEXT_DAWN_OWNED = 1u << 0,
  AURORA_VULKAN_NATIVE_CONTEXT_QUEUE_VERIFIED = 1u << 1,
};

/** One acquired OpenXR swapchain image for the next Aurora stereo sink. */
typedef struct {
  uint64_t image; /* VkImage */
  uint32_t width;
  uint32_t height;
  int32_t format; /* VkFormat */
} AuroraVulkanStereoTarget;

/**
 * Fired when Aurora either finishes or abandons the stereo sink. `success`
 * guarantees that the final same-queue Vulkan copy and its submission were
 * enqueued (and fenced) on Aurora's graphics queue. A false result may occur
 * after vkQueueSubmit, so callers must conservatively retain externally owned
 * targets until graphics/session teardown. The callback must not wait for the
 * GPU or re-enter Aurora.
 */
typedef void (*AuroraVulkanStereoSubmittedCallback)(uint64_t frameToken, bool success,
                                                    void* userdata);

/** Returns false unless the active Aurora backend is Dawn Vulkan. */
bool aurora_vulkan_get_native_handles(AuroraVulkanNativeContext* context);

/**
 * Installs the internal zero-readback stereo sink. Call while Aurora's frame
 * worker is idle, after aurora_initialize().
 */
bool aurora_vulkan_enable_stereo_bridge(AuroraVulkanStereoSubmittedCallback submitted,
                                        void* userdata);

/**
 * Publishes the acquired OpenXR image(s) for frameToken. Immersive projection
 * frames supply two targets; virtual-screen quad frames supply one. Exactly
 * one frame may be pending at a time.
 */
bool aurora_vulkan_set_stereo_targets(uint64_t frameToken,
                                      const AuroraVulkanStereoTarget* targets,
                                      uint32_t targetCount);

/**
 * Withdraws frameToken only while its target has not been encoded. This is
 * safe to race with Aurora's frame worker: false means the worker already owns
 * encoded work (or the token is no longer pending), so the submitted callback
 * remains the only completion authority. A successful cancellation performs
 * no GPU work and deliberately does not fire the callback.
 */
bool aurora_vulkan_cancel_stereo_targets(uint64_t frameToken);

/**
 * Removes the sink and drains bridge resources. The worker must be idle. Returns
 * false when queued work cannot be fenced; in that case the bridge is retained
 * for the process lifetime and the caller must likewise retain its graphics/XR
 * owners rather than destroying resources with unknown GPU use.
 */
bool aurora_vulkan_disable_stereo_bridge();

#ifdef __cplusplus
}
#endif

#endif