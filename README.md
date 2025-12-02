# ComfyUI-DGXSparkSafetensorsLoader
**A ComfyUI model loader that uses the [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) library to perform a very fast, zero-copy load from storage to VRAM.**

_This is very experimental, and may destroy the universe. So please don't use it in a production environment under any circumstances._

On DGX Spark, fastsafetensors is a massive improvement over the Hugging Face safetensors library for loading AI models. The Hugging Face library doesn't work well with the DGX Spark due to its architecture and memory design. Models load very slowly and sometimes use up to 2x memory during loading. This can cause large models to exceed the Spark's RAM capacity and fail, even when the model should fit in under half of the machine's RAM capacity.

This node doesn't require ComfyUI to be launched with the --cache-none or --disable-mmap options. The default options should work fine.

Here's an example of memory usage during and after loading the 60GB FLUX.2-dev BF16 model to the GPU and the 17GB Mistral FP8 text encoder to the CPU. As you can see, model loading happens extremely fast and memory usage never goes over 60%:

![FLUX.2-dev memory usage](https://github.com/user-attachments/assets/6a3a4ff7-bc4e-47ea-99b7-d4961b505a01)

# Known Issues
* Errors in yout ComfyUI workflow may leave the fastsafetensors loader in an allocated state, since ComfyUI can't free it. If this happens, you may get an OOM error on your next run. The workaround is to just quit and restart ComfyUI to clear VRAM.

* This node has not been tested on machines with discrete GPUs.
