import importlib
import sys

EXPECTED = {
    "JR_H3_DirectorDesk",
    "JR_MiniMaxH3AVLatentBuilder",
    "JR_H3_DirectedVideoConditioning",
    "JR_H3_OpenAICompatiblePromptOptimizer", "JR_H3_PromptReviewPause", "JR_H3_RTXUpscalerRefiner",
    "JR_H3_ResolutionScaleCalculator", "JR_H3_EnhancedVideoCombine", "JR_H3_LastFrame",
    "JR_H3_CacheConfigRouter", "JR_H3_AdaptiveCache",
    "JR_H3_UnifiedAcceleration",
    "JR_H3_HybridLoader",
    "JR_H3_TemporalChunkSampler",
}


def test_root_import_and_exact_registration(package_name):
    package = importlib.import_module(package_name)
    assert set(package.NODE_CLASS_MAPPINGS) == EXPECTED
    assert set(package.NODE_DISPLAY_NAME_MAPPINGS) == EXPECTED
    assert all(
        name.startswith(("JR MiniMax H3", "JR H3")) or name == "H3 Unified Acceleration"
        for name in package.NODE_DISPLAY_NAME_MAPPINGS.values()
    )
    assert package.__version__ == "0.11.0"
    assert package.WEB_DIRECTORY == "./js"


def test_rtx_dependency_is_lazy(package_name):
    sys.modules.pop("nvvfx", None)
    package = importlib.reload(importlib.import_module(package_name))
    assert "nvvfx" not in sys.modules
    assert "JR_H3_LastFrame" in package.NODE_CLASS_MAPPINGS


def test_modules_import_individually(package_name):
    for name in ["director_desk", "h3_av_latent_builder", "h3_directed_video_conditioning", "h3_hybrid_loader", "h3_openai_prompt_optimizer", "h3_temporal_chunk_sampler", "prompt_review_pause", "h3_cache_config_router", "h3_adaptive_cache", "h3_unified_acceleration", "rtx_upscaler_refiner", "resolution_scale_calculator", "enhanced_video_combine", "last_frame"]:
        importlib.import_module(f"{package_name}.nodes.{name}")
