"""JR MiniMax H3 custom nodes for ComfyUI."""

from .nodes.director_desk import JR_H3_DirectorDesk
from .nodes.enhanced_video_combine import JR_H3_EnhancedVideoCombine
from .nodes.h3_adaptive_cache import JR_H3_AdaptiveCache
from .nodes.h3_cache_config_router import JR_H3_CacheConfigRouter
from .nodes.h3_directed_video_conditioning import JR_H3_DirectedVideoConditioning
from .nodes.h3_openai_prompt_optimizer import JR_H3_OpenAICompatiblePromptOptimizer
from .nodes.h3_unified_acceleration import JR_H3_UnifiedAcceleration
from .nodes.last_frame import JR_H3_LastFrame
from .nodes.prompt_review_pause import JR_H3_PromptReviewPause
from .nodes.resolution_scale_calculator import JR_H3_ResolutionScaleCalculator
from .nodes.rtx_upscaler_refiner import JR_H3_RTXUpscalerRefiner
from .server import register_director_media_routes, register_prompt_review_routes

__version__ = "0.8.0"
WEB_DIRECTORY = "./js"

register_prompt_review_routes()
register_director_media_routes()

NODE_CLASS_MAPPINGS = {
    "JR_H3_DirectorDesk": JR_H3_DirectorDesk,
    "JR_H3_OpenAICompatiblePromptOptimizer": JR_H3_OpenAICompatiblePromptOptimizer,
    "JR_H3_DirectedVideoConditioning": JR_H3_DirectedVideoConditioning,
    "JR_H3_PromptReviewPause": JR_H3_PromptReviewPause,
    "JR_H3_CacheConfigRouter": JR_H3_CacheConfigRouter,
    "JR_H3_AdaptiveCache": JR_H3_AdaptiveCache,
    "JR_H3_UnifiedAcceleration": JR_H3_UnifiedAcceleration,
    "JR_H3_RTXUpscalerRefiner": JR_H3_RTXUpscalerRefiner,
    "JR_H3_ResolutionScaleCalculator": JR_H3_ResolutionScaleCalculator,
    "JR_H3_EnhancedVideoCombine": JR_H3_EnhancedVideoCombine,
    "JR_H3_LastFrame": JR_H3_LastFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JR_H3_DirectorDesk": "JR MiniMax H3 Director Desk",
    "JR_H3_OpenAICompatiblePromptOptimizer": "JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)",
    "JR_H3_DirectedVideoConditioning": "JR MiniMax H3 Directed Video Conditioning",
    "JR_H3_PromptReviewPause": "JR MiniMax H3 Prompt Review & Continue",
    "JR_H3_CacheConfigRouter": "JR H3 Cache Config Router",
    "JR_H3_AdaptiveCache": "JR H3 Adaptive Cache",
    "JR_H3_UnifiedAcceleration": "H3 Unified Acceleration",
    "JR_H3_RTXUpscalerRefiner": "JR MiniMax H3 RTX Upscaler & Refiner",
    "JR_H3_ResolutionScaleCalculator": "JR MiniMax H3 Resolution Scale Calculator",
    "JR_H3_EnhancedVideoCombine": "JR MiniMax H3 Enhanced Video Combine",
    "JR_H3_LastFrame": "JR MiniMax H3 Last Frame",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]
