"""
Util functions based on Diffuser framework.
"""


import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from diffusers import StableDiffusionPipeline, ControlNetModel


class MasaCtrlPipeline(StableDiffusionPipeline):
    
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]
    
    # prepare control image 
    def _prepare_control_image(            # ### NEW
        self,
        control_image,
        batch_size,
        height,
        width,
        device,
        do_classifier_free_guidance: bool,
    ):
        """
        把 PIL / ndarray / tensor 的 control 图变成
        (B 或 2B, 3, H, W)，范围 [-1, 1]，对齐 diffusers 0.15 的习惯
        """
        if control_image is None:
            return None

        import numpy as np
        import torch
        import torch.nn.functional as F

        if isinstance(control_image, Image.Image):
            control_image = np.array(control_image)

        if isinstance(control_image, np.ndarray):
            # HWC -> CHW
            control_image = torch.from_numpy(control_image).permute(2, 0, 1)

        # [0,255] -> [0,1]
        if control_image.dtype == torch.uint8:
            control_image = control_image.float() / 255.0
        else:
            control_image = control_image.float()

        # # [0,1] -> [-1,1]
        # control_image = control_image * 2.0 - 1.0

        # (3,H,W) -> (1,3,H,W)
        if control_image.ndim == 3:
            control_image = control_image.unsqueeze(0)

        control_image = F.interpolate(control_image, (height, width), mode="bilinear", align_corners=False)
        control_image = control_image.to(device)

        # CFG 情况下 batch 要翻倍
        if do_classifier_free_guidance:
            control_image = torch.cat([control_image] * 2, dim=0)

        return control_image

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        noise_loss_list=None,
        # controlnet condition
        control_image=None,
        controlnet_conditioning_scale=1.0,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        do_cfg = guidance_scale > 1.
        if do_cfg:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)


        # preprocess control image
        controlnet_cond = None                  # ### NEW
        if (self.controlnet is not None) and (control_image is not None):
            controlnet_cond = self._prepare_control_image(
                control_image,
                batch_size=batch_size,
                height=height,
                width=width,
                device=DEVICE,
                do_classifier_free_guidance=do_cfg,
            )

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            # CFG：latents 翻倍
            if do_cfg:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # 如果有 ControlNet，就走 ControlNet 分支
            if (self.controlnet is not None) and (controlnet_cond is not None):      # ### NEW
                # 保证 controlnet_cond 的 batch 维度和 model_inputs 对齐
                if controlnet_cond.shape[0] != model_inputs.shape[0]:
                    repeat = model_inputs.shape[0] // controlnet_cond.shape[0]
                    cond_for_step = controlnet_cond.repeat(repeat, 1, 1, 1)
                else:
                    cond_for_step = controlnet_cond

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    model_inputs,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=cond_for_step,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    model_inputs,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                # 原来的 no-controlnet 分支
                noise_pred = self.unet(
                    model_inputs,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

            # CFG 合成（原样）
            if do_cfg:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

            # scheduler step（原样）
            latents, pred_x0 = self.step(noise_pred, t, latents)
            if noise_loss_list is not None:
                latents = torch.concat((latents[:1]+noise_loss_list[i][:1], latents[1:]))
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image



        #     if guidance_scale > 1.:
        #         model_inputs = torch.cat([latents] * 2)
        #     else:
        #         model_inputs = latents
        #     if unconditioning is not None and isinstance(unconditioning, list):
        #         _, text_embeddings = text_embeddings.chunk(2)
        #         text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
        #     # predict the noise
        #     noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
        #     if guidance_scale > 1.:
        #         noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
        #         noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
        #     # compute the previous noise sample x_t -> x_t-1
        #     latents, pred_x0 = self.step(noise_pred, t, latents)
        #     if noise_loss_list is not None:
        #         latents = torch.concat((latents[:1]+noise_loss_list[i][:1],latents[1:]))
        #     latents_list.append(latents)
        #     pred_x0_list.append(pred_x0)

        # image = self.latent2image(latents, return_type="pt")
        # if return_intermediates:
        #     pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
        #     latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
        #     return image, pred_x0_list, latents_list
        # return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)
        # exit()
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
