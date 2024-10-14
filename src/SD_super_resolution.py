from diffusers import StableDiffusionUpscalePipeline


class Upsampler():
    def __init__(self):
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, variant="fp16", torch_dtype=torch.float16
                )
        pipeline = pipeline.to("cuda")

    def upscale(self, image, size, prompt):
        intermediate_size = (size[0] // 4, size[1] // 4)
        inter_image = image.resize(intermediate_size)
        upscaled_image = pipeline(prompt=prompt, image=inter_image).images[0]
        return upscaled_image

    def __call__(self, image, size, prompt):
        return self.upscale
