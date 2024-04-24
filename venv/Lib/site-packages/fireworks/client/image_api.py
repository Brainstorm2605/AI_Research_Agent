from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class Error(BaseModel):
    object: str = "error"
    type: str = "invalid_request_error"
    message: str


class ErrorResponse(BaseModel):
    error: Error = Field(default_factory=Error)


class TextPrompt(BaseModel):
    text: str
    weight: int


StyleLiteral = str


class StylePresetParams(BaseModel):
    style_preset: Optional[StyleLiteral] = None


class TextToImageRequestBody(StylePresetParams):
    model: str = "stable-diffusion-xl-1024-v1-0"
    cfg_scale: int = 7
    clip_guidance_preset: str = "NONE"
    height: int = 1024
    width: int = 1024
    sampler: Optional[str] = None
    samples: int = 1
    steps: int = 50
    seed: int = 0
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None

    # Deprecated - use `prompt` and `negative_prompt`
    text_prompts: Optional[List[TextPrompt]] = None

    safety_check: bool = False

    # LoRA parameters
    lora_adapter_name: Optional[str] = None
    lora_weight_filename: Optional[str] = None

    return_latents: bool = False


class ImageToImageRequestBody(StylePresetParams):
    model: str = "stable-diffusion-xl-1024-v1-0"
    cfg_scale: int = Field(default=7)
    clip_guidance_preset: str = "NONE"
    sampler: Optional[str]
    samples: int = 1
    steps: int = 50
    seed: int = 0

    init_image_mode: str = "IMAGE_STRENGTH"
    image_strength: Optional[float]
    step_schedule_start: Optional[float] = 0.65
    step_schedule_end: Optional[float]
    height: int = 1024
    width: int = 1024
    crop_padding: bool = True

    safety_check: bool = False
    return_latents: bool = False

    prompt: str = Field(default="")
    negative_prompt: Optional[str] = None

    # LoRA parameters
    lora_adapter_name: Optional[str] = None
    lora_weight_filename: Optional[str] = None

    @validator("image_strength", pre=True, always=True)
    def validate_image_strength(cls, v, values):
        if values["init_image_mode"] == "IMAGE_STRENGTH" and v is None:
            raise ValueError(
                "image_strength is required when init_image_mode is IMAGE_STRENGTH"
            )
        return v

    @validator("step_schedule_start", "step_schedule_end", pre=True, always=True)
    def validate_step_schedule(cls, v, values):
        if values["init_image_mode"] == "STEP_SCHEDULE" and v is None:
            raise ValueError(
                "Both step_schedule_start and step_schedule_end are required when init_image_mode is STEP_SCHEDULE"
            )
        return v


class ImageToVideoRequestBody(BaseModel):
    model: str = "stable-video-diffusion-img2vid"

    height: int = 576
    width: int = 1024
    sampler: Optional[str] = None
    frames: Optional[int] = None
    steps: int = 25
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 3.0
    fps: int = 7
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    decode_chunk_size: Optional[int] = None
    seed: int = 0
    safety_check: bool = False
    frame_interpolation_factor: Optional[int] = None
    output_video_bitrate: Optional[int] = None
    return_latents: bool = False

    infer_if_input_is_flagged_DEBUG_ONLY: bool = False


class ControlNetRequestBody(StylePresetParams):
    model: str = "stable-diffusion-xl-1024-v1-0"
    cfg_scale: int = Field(default=7)
    clip_guidance_preset: str = "NONE"
    sampler: Optional[str]
    samples: int = 1
    steps: int = 50
    seed: int = 0

    height: int = 1024
    width: int = 1024
    crop_padding: bool = True

    control_net_name: str
    conditioning_scale: float = 0.5

    step_schedule_start: Optional[float] = 0.0
    step_schedule_end: Optional[float] = 1.0

    safety_check: bool = False
    return_latents: bool = False

    prompt: str = Field(default="")
    negative_prompt: Optional[str] = None

    # LoRA parameters
    lora_adapter_name: Optional[str] = None
    lora_weight_filename: Optional[str] = None


class QRCodeRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
