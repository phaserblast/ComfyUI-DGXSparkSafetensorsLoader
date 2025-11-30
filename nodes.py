# https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader
# Copyright (c) 2025 Phaserblast. All rights reserved.
# Released under the Apache 2.0 license.
import torch
import folder_paths
import comfy.sd
import comfy.model_detection
import comfy.model_patcher

from safetensors import safe_open

# https://github.com/foundation-model-stack/fastsafetensors
from fastsafetensors import fastsafe_open,SafeTensorsFileLoader,SingleGroup

class DGXSparkSafetensorsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The filename of the .safetensors model to load."}),
				"dtype": (["fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"],{"default": "bf16"}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Node for loading a .safetensors file directly into memory using GPU Direct on DGX Spark."
    TITLE = "DGX Spark Safetensors Loader"
    
     
    def load_model(self, model_name, dtype):
        DTYPE_MAP = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        device = torch.device("cuda:0")
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        
        # Copy metadata only using HF safetensors library
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()
        
        # fastsafetensors
        loader = SafeTensorsFileLoader(SingleGroup(), device)
        loader.add_filenames({0: [model_path]})
        fb = loader.copy_files_to_device()
        keys = list(fb.key_to_rank_lidx.keys())
        sd = {} # state dictionary
        for k in keys:
            sd[k] = fb.get_tensor(k)
        #fb.close() # No!
        #loader.close() # No!
        
        # You can also try with fastsafe_open instead of SafeTensorsFileLoader.
        # This might work just as well.
        # with fastsafe_open(filenames=[model_path], nogds=False, device=device, debug_log=True) as f:
        #    sd = {} # state dictionary
        #    for k in f.keys():
        #        sd[k] = f.get_tensor(k)
        #        #sd[k] = f.get_tensor(k).detach().clone() # clone if tensor is used outside (uses 2x memory!)
		#        #sd[k] = f.get_tensor(k).to(device=dev, copy=False) # uses 2x memory!
	
        # Init the model to pass to ComfyUI
        model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
        model_config.set_inference_dtype(DTYPE_MAP[dtype], DTYPE_MAP[dtype])
        model = model_config.get_model(sd, "", device=None)
        
        # Use this instead of load_model_weights()
        # I think 'diffusion_model' is a subclass of torch.nn.Module,
        # so we can use assign=True in load_state_dict which avoids
        # a copy. Just make sure you don't free the tensors read
        # by fastsafetensors if we use this option.
        #
        # The following duplicates the functionality of load_model_weights():
        #
        # Ensure stuff isn't duplicated on the GPU
        model = model.to(None)
        #
        sd = model_config.process_unet_state_dict(sd)
        # load_state_dict is from torch.nn.Module
        # If using assign=True, then don't free the tensors
        # loaded with fastsafetensors.
        model.diffusion_model.load_state_dict(sd, strict=False, assign=True)
        
        model = comfy.model_patcher.ModelPatcher(model, load_device=device, offload_device=None)
        
        # Don't free anything, this is just here for completeness 
        #del sd
        #fb.close()
        #loader.close()
        return (model,)

