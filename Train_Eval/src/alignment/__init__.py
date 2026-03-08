__version__ = "0.3.0.dev0"

from .configs import DataArguments, DPOConfig, H4ArgumentParser, ModelArguments, SFTConfig, RMConfig
from .data import apply_chat_template, get_datasets
# 延迟导入decontamination，避免网络问题
try:
    from .decontaminate import decontaminate_humaneval
except Exception:
    decontaminate_humaneval = None
from .model_utils import (
    get_checkpoint,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)


__all__ = [
    "DataArguments",
    "DPOConfig",
    "H4ArgumentParser",
    "ModelArguments",
    "SFTConfig",
    "RMConfig",
    "apply_chat_template",
    "get_datasets",
    "get_checkpoint",
    "get_kbit_device_map",
    "get_peft_config",
    "get_quantization_config",
    "get_tokenizer",
    "is_adapter_model",
]

# 只有在成功导入时才添加到__all__
if decontaminate_humaneval is not None:
    __all__.append("decontaminate_humaneval")
