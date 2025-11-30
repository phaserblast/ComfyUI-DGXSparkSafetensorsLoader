# ComfyUI-DGXSparkSafetensorsLoader
**A ComfyUI model loader that uses the fastsafetensors library to perform a very fast, zero-copy load from storage to VRAM.**

This is very experimental, and may destroy the universe. So please don't use it in a production environment under any circumstances.

Here's an example of memory usage during and after loading the 60GB FLUX.2-dev BF16 model to the GPU and the 17GB Mistral FP8 text encoder to the CPU. As you can see, model loading happens extremely fast and memory usage never goes over 60%:

![FLUX.2-dev memory usage](https://github.com/user-attachments/assets/6a3a4ff7-bc4e-47ea-99b7-d4961b505a01)

fastsafetensors library project on GitHub:
https://github.com/foundation-model-stack/fastsafetensors
