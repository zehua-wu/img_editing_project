import torch

from models.p2p.attention_control import register_attention_control
from utils.utils import init_latent

def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents



@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent

@torch.no_grad()
def p2p_guidance_forward_single_branch(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([torch.cat([uncond_embeddings[i],uncond_embeddings_[1:]]), text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]+noise_loss[1:]))
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def direct_inversion_p2p_guidance_forward(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
    return latents, latent

@torch.no_grad()
def p2p_guidance_forward_ctrl(
    model,             # StableDiffusionPipeline
    controlnet,        # ControlNetModel
    control_hint,      # [B, 3, 512, 512] tensor . B = 1?
    prompt,
    controller,
    latent,
    num_inference_steps,
    guidance_scale,
    generator=None,
    uncond_embeddings=None,
    control_scale=1.0,
):
    
    from models.p2p.attention_control import register_attention_control
    from utils.utils import init_latent

    device = model.device
    batch_size = len(prompt)
    height = width = 512

    # must reg!
    register_attention_control(model, controller)
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]

    # unconditional embeddings!!!!
    #  2 situations
    #  null-text inversion: uncond_embeddings is a ' len=T ' list:with each [1, L, C]
    #  common: uncond_embeddings None, we make a fixed one
    use_per_step_uncond = isinstance(uncond_embeddings, (list, tuple))

    if not use_per_step_uncond:
        # normal cfg, direct unconditional embedding
        uncond_input = model.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_base = model.text_encoder(uncond_input.input_ids.to(device))[0]
    else:
        uncond_base = None  # every step get from list

    #  init latents：expand into [batch_size, C, H/8, W/8]!!
    latent, latents = init_latent(
        latent=latent,
        model=model,
        height=height,
        width=width,
        generator=generator,
        batch_size=batch_size,
    )

    # ddim
    model.scheduler.set_timesteps(num_inference_steps)
    timesteps = model.scheduler.timesteps

    for i, t in enumerate(timesteps):
        # this timestep unconditional embedding
        if use_per_step_uncond:
            # uncond_embeddings[i]: [1, L, C] -> [B, L, C]
            uncond_step = uncond_embeddings[i].expand(batch_size, -1, -1)
        else:
            uncond_step = uncond_base

        # context: [2B, L, C]  (first B uncond，last B cond)
        context = torch.cat([uncond_step, text_embeddings], dim=0)

        # cfg：latents -> [2B, C, H, W]
        latents_input = torch.cat([latents] * 2, dim=0)

        # ControlNet forward
        # [1, 3, 512, 512] or [B, 3, 512, 512]
        if control_hint.shape[0] == 1:
            ctrl_in = control_hint.expand(latents_input.shape[0], -1, -1, -1)
        else:
            # if [B,3,H ,  W]，then [2B,3 ,H,W]
            ctrl_in = control_hint.expand(2 * batch_size, -1, -1, -1)

        ctrl_out = controlnet(
            latents_input,
            t,
            encoder_hidden_states=context,
            controlnet_cond=ctrl_in,
            return_dict=True,
        )
        down_res = [r * control_scale for r in ctrl_out.down_block_res_samples]
        mid_res = ctrl_out.mid_block_res_sample * control_scale

        # UNet + P2P attention controller
        noise_pred = model.unet(
            latents_input,
            t,
            encoder_hidden_states=context,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            cross_attention_kwargs={"controller": controller},
        ).sample

        # cfg
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        latents = model.scheduler.step(noise, t, latents).prev_sample

        # controller can onestep callback
        latents = controller.step_callback(latents)

    return latents, latent