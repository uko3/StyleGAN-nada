# modules/trainer.py

import torch
import torch.optim as optim
import copy
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import clip
import os
from sklearn.decomposition import PCA
import warnings

# warnings.filterwarnings("ignore", message="conv2d_gradfix not supported on PyTorch.*") # Already handled in Colab setup

# Импортируем классы из других модулей
#from modules.model import Generator
#from modules.losses import CLIPLoss, CLIPDirectionalLoss
#from modules.utils import freeze_layers_adaptive # Будет создан ниже

class LatentStyleTrainer:
    def __init__(
        self,
        generator,
        model_clip,
        text_features_source,
        text_features_target,
        freeze_fn,
        clip_directional_loss,
        latent_dim,
        batch_size,
        device,
        lr_generator=0.001,
        lr_lambda=0.2,
        weight_decay=0.0,
        lambda_clip_init=1.0,
        lambda_l2_init=1.0,
    ):
        self.device = device
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.freeze_fn = freeze_fn
        self.clip_loss_fn = clip_directional_loss
        self.model_clip = model_clip
        self.lr_generator = lr_generator
        self.weight_decay = weight_decay
        self.model = {
            "generator_frozen": copy.deepcopy(generator).to(device).eval(),
            "generator_train": copy.deepcopy(generator).to(device).train()
        }

        # Ensure generator_frozen parameters are actually frozen
        for param in self.model["generator_frozen"].parameters():
            param.requires_grad = False

        self.lambda_t = torch.tensor(
            [math.log(lambda_clip_init), math.log(lambda_l2_init)],
            device=device,
            requires_grad=True
        )

        self.optimizer_lambda = torch.optim.Adam([self.lambda_t], lr=lr_lambda)
        # Optimizer for generator will be set after freezing in train method
        self.optimizer_generator = None

        self.text_target = text_features_target
        self.text_source = text_features_source

        self.losses = {'l2': [], 'clip': [], 'all': []}

    def sample_latent_w(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        latent_z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        with torch.no_grad():
            latent_w = self.model['generator_frozen'].style(latent_z)
        return latent_w

    def train(self, epochs, freeze_each_epoch=True, reclassify=False, seed=None):
        # Initial freezing outside the loop if not freezing each epoch
        if not freeze_each_epoch:
            self.freeze_fn(
                self.model['generator_train'],
                self.model['generator_frozen'],
                self.text_target, # Pass text_target to freeze_layers_adaptive
                top_k=10 # Example k value
            )
            # Initialize optimizer_generator AFTER initial freezing
            self.optimizer_generator = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model["generator_train"].parameters()),
                lr=self.lr_generator, # Use self.lr_generator from init
                weight_decay=self.weight_decay # Use self.weight_decay from init
            )
        else: # If freezing each epoch, initialize optimizer with all params initially
             self.optimizer_generator = torch.optim.Adam(
                self.model["generator_train"].parameters(), # All params initially
                lr=self.lr_generator,
                weight_decay=self.weight_decay
            )


        for epoch in range(epochs):
            torch.cuda.empty_cache()
            self.optimizer_generator.zero_grad()
            self.optimizer_lambda.zero_grad()

            if freeze_each_epoch:
                self.freeze_fn(
                    self.model['generator_train'],
                    self.model['generator_frozen'],
                    self.text_target, 
                    top_k=10 
                )
                # Re-initialize optimizer_generator with newly unfrozen parameters if freezing each epoch
                # This ensures the optimizer only works on active parameters
                self.optimizer_generator = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model["generator_train"].parameters()),
                    lr=self.lr_generator,
                    weight_decay=self.weight_decay
                )

            latent_w = self.sample_latent_w(seed=seed)

            generated_img_frozen, _ = self.model['generator_frozen']([latent_w], input_is_latent=True, randomize_noise=False)
            generated_img_style, _ = self.model['generator_train']([latent_w], input_is_latent=True, randomize_noise=False)

            if reclassify:
               img = (generated_img_frozen[0].detach().cpu().clamp(-1, 1) + 1) / 2
               img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
               source_class = clip_classification(img_pil, text_features_cat)
               text_source_clp = clip.tokenize([source_class]).to(self.device)
               with torch.no_grad():
                   self.text_source = self.model_clip.encode_text(text_source_clp)
                   self.text_source = self.text_source / self.text_source.norm(dim=-1, keepdim=True)

            lambda_clip_val = torch.exp(self.lambda_t[0])
            lambda_l2_val = torch.exp(self.lambda_t[1])

            clip_loss_val = self.clip_loss_fn(generated_img_frozen, generated_img_style, self.text_source, self.text_target)
            l2_loss_val = F.mse_loss(generated_img_style, generated_img_frozen)
            loss_total = lambda_clip_val * clip_loss_val + lambda_l2_val * l2_loss_val

            loss_total.backward()
            self.optimizer_generator.step()
            self.optimizer_lambda.step()

            self.losses['l2'].append(l2_loss_val.item())
            self.losses['clip'].append(clip_loss_val) # clip_loss_val is already a scalar from .item() in CLIPDirectionalLoss
            self.losses['all'].append(loss_total.item())

            print(f"[{epoch}/{epochs}] Loss: {loss_total.item():.4f} | CLIP: {clip_loss_val:.4f} | L2: {l2_loss_val.item():.4f}")

            if epoch % 10 == 0:
                self.visualize_images(generated_img_frozen, generated_img_style, epoch)


    def plot_losses(self):
        plt.figure(figsize=(7, 4))
        plt.plot(self.losses['l2'], label='L2 Loss')
        plt.plot(self.losses['clip'], label='CLIP Directional Loss')
        plt.plot(self.losses['all'], label='Total Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_images(self, generated_img_frozen, generated_img_style, epoch):
        batch_size = generated_img_frozen.size(0)
        plt.figure(figsize=(batch_size * 2, 4))

        for i in range(batch_size):
            img = (generated_img_frozen[i].detach().cpu().clamp(-1, 1) + 1) / 2
            img = img.permute(1, 2, 0).numpy()
            plt.subplot(2, batch_size, i + 1)
            plt.imshow(img)
            plt.title(f"Frozen {i}")
            plt.axis('off')

        for i in range(batch_size):
            img = (generated_img_style[i].detach().cpu().clamp(-1, 1) + 1) / 2
            img = img.permute(1, 2, 0).numpy()
            plt.subplot(2, batch_size, batch_size + i + 1)
            plt.imshow(img)
            plt.title(f"Styled {i}")
            plt.axis('off')

        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def visualize_clip_directions(self, image_frozen, image_styled, text_target, text_source, preprocess):
        img_frozen = Image.fromarray(((image_frozen[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8))
        img_style = Image.fromarray(((image_styled[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8))

        clip_in_frozen = preprocess(img_frozen).unsqueeze(0).to(self.device)
        clip_in_styled = preprocess(img_style).unsqueeze(0).to(self.device)

        feat_img_froz = self.model_clip.encode_image(clip_in_frozen)
        feat_img_style = self.model_clip.encode_image(clip_in_styled)
        feat_text_src = self.model_clip.encode_text(text_source)
        feat_text_tgt = self.model_clip.encode_text(text_target)

        all_vectors = torch.cat([feat_img_froz, feat_img_style, feat_text_src, feat_text_tgt], dim=0)
        vec_2d = PCA(n_components=2).fit_transform(all_vectors.cpu().numpy())

        i_froz, i_style, t_src, t_tgt = vec_2d

        plt.figure(figsize=(4, 4))
        plt.quiver(i_froz[0], i_froz[1], i_style[0] - i_froz[0], i_style[1] - i_froz[1],
                   angles='xy', scale_units='xy', scale=1, color='blue', label='Image Direction')
        plt.quiver(t_src[0], t_src[1], t_tgt[0] - t_src[0], t_tgt[1] - t_src[1],
                   angles='xy', scale_units='xy', scale=1, color='red', label='Text Direction')

        plt.scatter([i_froz[0], i_style[0]], [i_froz[1], i_style[1]], color='blue', label='Image Points')
        plt.scatter([t_src[0], t_tgt[0]], [t_src[1], t_tgt[1]], color='red', label='Text Points')
        plt.legend()
        plt.title("CLIP Embedding Directions (2D PCA)")
        plt.grid(True)
        plt.show()
