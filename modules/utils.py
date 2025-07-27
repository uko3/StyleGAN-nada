import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
#import clip
import math
from sklearn.decomposition import PCA
import dlib 
from torchvision import transforms 
from modules.losses import CLIPLoss 
import re


def freeze_layers_adaptive(model_train, model_frozen, text_target_features, k=5, auto_layer_iters=3, device='cuda'):
    batch_size_temp = 2
    latent_dim = model_frozen.style_dim

    latent_z_temp = torch.randn(batch_size_temp, latent_dim, device=device)
    with torch.no_grad():
        latent_w_temp = model_frozen.style(latent_z_temp)
    latent_w_plus_temp = latent_w_temp.unsqueeze(1).repeat(1, model_frozen.n_latent, 1)

    latent_tensor= torch.Tensor(latent_w_plus_temp.cpu().detach().numpy()).to(device)
    latent_tensor.requires_grad = True

    fl_optimizer = torch.optim.Adam([latent_tensor], lr=0.01)

    clip_loss_for_freezing = CLIPLoss(stylegan_size=model_train.size) 

    for i_iter in range(auto_layer_iters):
        fl_optimizer.zero_grad()
        generated_img_fl, _ = model_train([latent_tensor], input_is_latent=True)
 
        selection_loss = clip_loss_for_freezing(generated_img_fl, text_target_features)
        selection_loss.backward()
        fl_optimizer.step()

    involved_layers = torch.abs(latent_tensor - latent_w_plus_temp).mean(dim=-1).mean(dim=0)
    used_layers = torch.topk(involved_layers, k)[1].cpu().numpy()

    all_children = list(model_train.children())
    potential_layers = [
        all_children[2],  
        all_children[3]   
    ]
    potential_layers.extend(list(all_children[4]))

    used_modules = [potential_layers[idx] for idx in used_layers]

    named_modules = list(model_train.named_modules())
    used_layer_names = [name for name, module in named_modules if module in used_modules] 

    for param in model_train.parameters():
        param.requires_grad = False

    for name, param in model_train.named_parameters():
        if any(name.startswith(used_name) for used_name in used_layer_names):
             param.requires_grad = True



def generate_visualize_and_save(
    trainer,
    seeds,
    output_dir="generated_val_images",
    folder_name="styled_only"
):
    save_dir = os.path.join(output_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    row1 = []  # frozen
    row2 = []  # styled

    for i, seed in enumerate(seeds):
        latent_w = trainer.sample_latent_w(seed=seed)

        with torch.no_grad():
            img_frozen, _ = trainer.model["generator_frozen"]([latent_w], input_is_latent=True, randomize_noise=False)
            img_styled, _ = trainer.model["generator_train"]([latent_w], input_is_latent=True, randomize_noise=False)

        img_frozen = (img_frozen.clamp(-1, 1) + 1) / 2
        img_styled = (img_styled.clamp(-1, 1) + 1) / 2

        row1.append(img_frozen[0].cpu())
        row2.append(img_styled[0].cpu())

        img = img_styled[0].cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype('uint8')
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(save_dir, f"styled_seed_{seed}.png"))

    ncols = len(seeds)
    fig, axs = plt.subplots(2, ncols, figsize=(ncols * 2, 5))

    for i in range(ncols):
        axs[0, i].imshow(row1[i].permute(1, 2, 0).numpy())
        axs[0, i].axis('off')
        axs[0, i].set_title(f"{i+1}", fontsize=10)

        axs[1, i].imshow(row2[i].permute(1, 2, 0).numpy())
        axs[1, i].axis('off')

    fig.text(0.05, 0.76, "Frozen", va='center', ha='right', fontsize=14)
    fig.text(0.05, 0.29, "Styled", va='center', ha='right', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    plt.show()

    print(f"Saved styled images to: {save_dir}")

