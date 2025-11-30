# https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader
# Copyright (c) 2025 Phaserblast. All rights reserved.
# Released under the Apache 2.0 license.
from .nodes import *

NODE_CLASS_MAPPINGS = {
	"DGXSparkSafetensorsLoader": DGXSparkSafetensorsLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
	"DGXSparkSafetensorsLoader": "DGX Spark Safetensors Loader",
}

