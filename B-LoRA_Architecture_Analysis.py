try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is not installed. Please install the dependencies in requirements.txt"
    ) from e

from diffusers import StableDiffusionXLPipeline

def inject_cross_attention_per_layer(pipe, embeddings_map: dict):
    handles = []

    def make_hook(embeddings):
        def hook(module, input_tuple):
            input_list = list(input_tuple)
            if len(input_list) > 1:
                input_list[1] = embeddings
            return tuple(input_list)
        return hook

    for name, module in pipe.unet.named_modules():
        for key in embeddings_map.keys():
            if key in name and "attn2" in name:
                embeddings = embeddings_map[key]
                handle = module.register_forward_pre_hook(make_hook(embeddings))
                handles.append(handle)
                print(f"Hook registered for {name}")

    return handles


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SDXL pipeline をロード
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16", 
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # プロンプト設定
    p_content = "A photo of a bunny"
    phat_content = "A photo of a tiger"


    # 埋め込み生成
    (
        prompt_embeds_p,
        negative_prompt_embeds_p,
        pooled_prompt_embeds_p,
        negative_pooled_prompt_embeds_p,
    ) = pipe.encode_prompt(
        prompt=p_content,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    (
        prompt_embeds_p_hat,
        negative_prompt_embeds_p_hat,
        pooled_prompt_embeds_p_hat,
        negative_pooled_prompt_embeds_p_hat,
    ) = pipe.encode_prompt(
        prompt=phat_content,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )


    # p^ を特定の層にだけ注入，それ以外は p
    embeddings_map = {
        "down_blocks.2.attentions.0": prompt_embeds_p,
        "down_blocks.2.attentions.1": prompt_embeds_p_hat,
        "mid_block.attentions.0": prompt_embeds_p,
        "up_blocks.0.attentions.0": prompt_embeds_p,
        "up_blocks.0.attentions.1": prompt_embeds_p,
        "up_blocks.0.attentions.2": prompt_embeds_p,
    }
    
    dummy_prompt = ""  # 中身なし
    (
        dummy_prompt_embeds,
        dummy_negative_prompt_embeds,
        dummy_pooled_prompt_embeds,
        dummy_negative_pooled_prompt_embeds
    ) = pipe.encode_prompt(
        prompt=dummy_prompt,
        device=device,
        do_classifier_free_guidance=True,
    )

    # Hook登録 & 推論
    with torch.no_grad():
        handles = inject_cross_attention_per_layer(pipe, embeddings_map)
        print("Registered hooks for:")
        for h in handles:
            print(f"  {h}")

        output = pipe(
            prompt_embeds=dummy_prompt_embeds, # 空のプロンプトを渡す
            negative_prompt_embeds=dummy_negative_prompt_embeds,
            pooled_prompt_embeds=dummy_pooled_prompt_embeds,
            negative_pooled_prompt_embeds=dummy_negative_pooled_prompt_embeds,
            guidance_scale=7.5,
            num_inference_steps=30,
            # generator=torch.Generator(device=device).manual_seed(42),
        )


        for h in handles:
            h.remove()

    # 出力画像保存
    image = output.images[0]
    image.save("output.png")


if __name__ == "__main__":
    main()