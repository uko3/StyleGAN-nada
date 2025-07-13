import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import clip
import math
from sklearn.decomposition import PCA
import dlib 
from torchvision import transforms 
from modules.losses import CLIPLoss 
import re
from modules.stylegan_arch.model import StyledConv, ToRGB, Generator


def freeze_layers_adaptive(model_train, model_frozen, text_target_features, k=5, auto_layer_iters=3, device='cuda'):

    batch_size_temp = 2
    latent_dim = model_frozen.style_dim

    latent_z_temp = torch.randn(batch_size_temp, latent_dim, device=device)
    with torch.no_grad():
        latent_w_temp = model_frozen.style(latent_z_temp)
    latent_w_plus_temp = latent_w_temp.unsqueeze(1).repeat(1, model_frozen.n_latent, 1)

 
    w_codes_optimizable = latent_w_plus_temp.clone().detach().requires_grad_(True)
    w_optim_selection = torch.optim.Adam([w_codes_optimizable], lr=0.01)


    clip_loss_for_freezing = CLIPLoss(stylegan_size=model_train.size) 

    for i_iter in range(auto_layer_iters):
        w_optim_selection.zero_grad()
        generated_for_selection, _ = model_train([w_codes_optimizable], input_is_latent=True)
 
        selection_loss = clip_loss_for_freezing(generated_for_selection, text_target_features)
        selection_loss.backward()
        w_optim_selection.step()

    layer_weights = torch.abs(w_codes_optimizable - latent_w_plus_temp).mean(dim=-1).mean(dim=0)
    chosen_layer_idx = torch.topk(layer_weights, k)[1].cpu().numpy()

    all_children = list(model_train.children())
    potential_layers = []

    named_modules_list = [(name, module) for name, module in model_train.named_modules()]
    
    w_plus_module_map = []

    w_plus_module_map.append((0, 'conv1'))
    w_plus_module_map.append((1, 'to_rgb1'))

    prefix_map = {
        0: 'conv1',
        1: 'to_rgb1'
    }

    named_modules_dict = dict(model_train.named_modules())
    
    potential_style_modules = []
    
    if 'conv1' in named_modules_dict:
        potential_style_modules.append(named_modules_dict['conv1'])
    if 'to_rgb1' in named_modules_dict:
        potential_style_modules.append(named_modules_dict['to_rgb1'])
        
    for name, module in named_modules_dict.items():
        if re.match(r'convs\.\d+', name) or re.match(r'to_rgbs\.\d+', name):
            if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, (StyledConv, ToRGB)):
                        potential_style_modules.append(sub_module)
            elif isinstance(module, (StyledConv, ToRGB)): 
                potential_style_modules.append(module)

    w_idx_to_module_names = {}
    
    # W+ index 0
    w_idx_to_module_names[0] = ['conv1']
    # W+ index 1
    w_idx_to_module_names[1] = ['to_rgb1']

   
    for i_stage in range(model_train.n_latent // 2 - 1): 

        w_idx_first_conv = i_stage * 2 + 2
        w_idx_to_module_names.setdefault(w_idx_first_conv, []).append(f'convs.{i_stage*2}')
        
        w_idx_second_conv = i_stage * 2 + 3
        w_idx_to_module_names.setdefault(w_idx_second_conv, []).append(f'convs.{i_stage*2+1}')
        w_idx_to_module_names.setdefault(w_idx_second_conv, []).append(f'to_rgbs.{i_stage}')

    actual_modules_to_unfreeze_base_names = set()
    for w_idx in chosen_layer_idx:
        if w_idx in w_idx_to_module_names:
            for name_prefix in w_idx_to_module_names[w_idx]:
                actual_modules_to_unfreeze_base_names.add(name_prefix)

    for param in model_train.parameters():
        param.requires_grad = False

    for name, param in model_train.named_parameters():
        if any(name.startswith(base_name) for base_name in actual_modules_to_unfreeze_base_names):
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

