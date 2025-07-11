# modules/utils.py

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import clip
import math
from sklearn.decomposition import PCA
import dlib # Для выравнивания лица
from torchvision import transforms # Для предобработки изображений для инверсии
from model import Generator # Для pSp
from losses import CLIPLoss # Для freeze_layers_adaptive

# Assuming pSp model is defined elsewhere or will be passed in
# from model import pSp # If pSp is a custom model

# Placeholder for align_face if not provided by StyleGAN2 directly
# This is usually from a separate utils file or specific alignment library
# For now, I'll put a placeholder. You need to ensure 'utils.alignment' exists or provide the function.
def align_face(filepath, predictor):
    """
    Placeholder for face alignment function.
    You need to provide your actual implementation.
    """
    # Example dummy implementation:
    # This would typically use dlib or similar libraries to find and align the face.
    # For now, it just opens the image.
    img = Image.open(filepath).convert("RGB")
    # In a real scenario, you would perform face detection and alignment here
    # For demonstration, we'll just return the original image resized
    print(f"Warning: Using dummy align_face. Replace with actual implementation if needed for alignment.")
    return img.resize((1024, 1024)) # Or whatever target size


def freeze_layers_adaptive(model_train, model_frozen, text_target_features, k=5, auto_layer_iters=3, device='cuda'):
    print("Applying adaptive layer freezing...")
    batch_size_temp = 2
    latent_dim = model_frozen.latent_dim

    latent_z_temp = torch.randn(batch_size_temp, latent_dim, device=device)
    with torch.no_grad():
        latent_w_temp = model_frozen.style(latent_z_temp)
    latent_w_plus_temp = latent_w_temp.unsqueeze(1).repeat(1, model_frozen.n_latent, 1)

    # Make w_codes optimizable for layer selection
    w_codes_optimizable = latent_w_plus_temp.clone().detach().requires_grad_(True)
    w_optim_selection = torch.optim.Adam([w_codes_optimizable], lr=0.01)

    # Initialize a CLIPLoss instance for this function scope
    clip_loss_for_freezing = CLIPLoss(stylegan_size=model_train.size) # Pass generator size to CLIPLoss

    for i_iter in range(auto_layer_iters):
        w_optim_selection.zero_grad()
        generated_for_selection, _ = model_train([w_codes_optimizable], input_is_latent=True)
        # Use the CLIPLoss instance with normalized text features
        selection_loss = clip_loss_for_freezing(generated_for_selection, text_target_features)
        selection_loss.backward()
        w_optim_selection.step()
        # print(f"  Selection Iteration {i_iter+1}/{auto_layer_iters}, Loss: {selection_loss.item():.4f}") # Uncomment for verbose debugging

    layer_weights = torch.abs(w_codes_optimizable - latent_w_plus_temp).mean(dim=-1).mean(dim=0)
    chosen_layer_idx = torch.topk(layer_weights, k)[1].cpu().numpy()
    print(f"  Chosen W+ indices to unfreeze: {chosen_layer_idx}")

    all_children = list(model_train.children())
    potential_layers = []
    # Identify layers that receive W+
    # Adjust indexing based on your Generator's __init__
    # Example:
    # self.input (index 0) - not modulated by W+
    # self.conv1 (index 1) - takes style_codes[:, 0]
    # self.to_rgb1 (index 2) - takes style_codes[:, 1]
    # self.convs (ModuleList, index 3) - takes style_codes[:, 2] to style_codes[:, 17]
    # self.to_rgbs (ModuleList, index 4) - takes style_codes (skip connections)

    # Corrected based on your Generator (rosinality's):
    # style (index 0)
    # input (index 1)
    # conv1 (index 2) -> W+ index 0
    # to_rgb1 (index 3) -> W+ index 1
    # convs (index 4) -> W+ indices 2 to 17 (8 pairs of StyledConv)
    # to_rgbs (index 5) -> W+ indices based on convs

    # The W+ indices (0-17) correspond to:
    # 0: conv1
    # 1: to_rgb1
    # 2: convs[0]
    # 3: convs[1]
    # ...
    # 16: convs[14]
    # 17: convs[15]

    # Let's map these properly
    # Using regex to find the corresponding module names for the chosen W+ indices
    # This is more robust than fixed indices if model architecture changes slightly
    named_modules_list = [(name, module) for name, module in model_train.named_modules()]
    
    # Store tuples of (w_plus_index, module_name)
    w_plus_module_map = []

    # Map for conv1 and to_rgb1
    w_plus_module_map.append((0, 'conv1'))
    w_plus_module_map.append((1, 'to_rgb1'))

    # Map for convs and to_rgbs
    # Based on StyleGAN2's structure: convs are StyledConv, to_rgbs are ToRGB
    # Each StyledConv takes a style code. ToRGB also takes one.
    # The actual W+ index mapping needs to be accurate.
    # From your Generator class:
    # style_codes[:, 0] -> conv1
    # style_codes[:, 1] -> to_rgb1
    # style_codes[:, i * 2 + 2] -> convs[i*2] (first conv in pair)
    # style_codes[:, i * 2 + 3] -> convs[i*2+1] (second conv in pair) AND to_rgbs[i]

    # This means W+ indices are tied to specific modules as follows:
    # W+ index 0: conv1
    # W+ index 1: to_rgb1
    # W+ index 2: convs.0 (StyledConv)
    # W+ index 3: convs.1 (StyledConv) AND to_rgbs.0
    # W+ index 4: convs.2 (StyledConv)
    # W+ index 5: convs.3 (StyledConv) AND to_rgbs.1
    # ... and so on.

    # So if you select W+ index 3, you want to unfreeze convs.1 and to_rgbs.0.
    # This becomes tricky because one W+ index can affect multiple modules or one module can be affected by multiple W+ indices.
    # The current `freeze_layers_adaptive` in the previous code snippet simplifies this by saying:
    # "if module in selected_modules_to_unfreeze" where selected_modules_to_unfreeze are from `potential_layers`.

    # Let's refine `potential_layers` and `used_layer_names` to be more direct.
    # We want to unfreeze modules whose style input index matches the chosen W+ index.
    
    # Mapping W+ indices to module prefixes for `named_parameters()`
    prefix_map = {
        0: 'conv1',
        1: 'to_rgb1'
    }
    # For convs.X and to_rgbs.Y
    # convs[i*2] takes style_codes[:, i*2+2]
    # convs[i*2+1] takes style_codes[:, i*2+3]
    # to_rgbs[i] takes style_codes[:, i*2+3]
    
    # Max n_latent for size 1024 is 18 (0-17)
    # i goes from 0 to 7 for convs (16 convs total)
    # i goes from 0 to 7 for to_rgbs (8 to_rgbs total)

    # For `convs` (StyledConv modules, which contain ModulatedConv2d)
    # and `to_rgbs` (ToRGB modules, which contain ModulatedConv2d)
    # The `ModulatedConv2d` inside them has `modulation.weight` and `modulation.bias`
    # and `conv.weight`, `conv.bias`.

    # Rebuilding `potential_layers` to contain module objects that directly receive style codes
    # This is more direct than relying on `all_children` fixed indices.
    
    # Get the named modules directly:
    named_modules_dict = dict(model_train.named_modules())
    
    potential_style_modules = []
    
    # Add conv1 and to_rgb1
    if 'conv1' in named_modules_dict:
        potential_style_modules.append(named_modules_dict['conv1'])
    if 'to_rgb1' in named_modules_dict:
        potential_style_modules.append(named_modules_dict['to_rgb1'])
        
    # Add modules from convs and to_rgbs ModuleLists
    # Assuming the W+ indices are consistently assigned
    for name, module in named_modules_dict.items():
        if re.match(r'convs\.\d+', name) or re.match(r'to_rgbs\.\d+', name):
            if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                # If it's a list/sequential, iterate its children if they are the actual styled modules
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, (StyledConv, ToRGB)): # Use your actual StyledConv/ToRGB
                        potential_style_modules.append(sub_module)
            elif isinstance(module, (StyledConv, ToRGB)): # Direct StyledConv/ToRGB
                potential_style_modules.append(module)

    # Filter `potential_style_modules` based on `chosen_layer_idx` by their logical W+ index.
    # This requires knowing the W+ index associated with each module.
    # This is the most complex part of `freeze_layers_adaptive`.

    # Simpler approach: Map `chosen_layer_idx` back to names.
    # The `involved_layers` calculation correctly gives you the indices of W+ that changed most.
    # We need to correctly map these W+ indices (0-17) to the actual module names that consume them.
    
    # The `Generator` class style input mapping:
    # `style_codes[:, 0]` -> `conv1`
    # `style_codes[:, 1]` -> `to_rgb1`
    # `style_codes[:, i * 2 + 2]` -> `convs[i*2]`
    # `style_codes[:, i * 2 + 3]` -> `convs[i*2+1]` and `to_rgbs[i]`

    # Let's create a mapping from W+ index to the module names it affects.
    w_idx_to_module_names = {}
    
    # W+ index 0
    w_idx_to_module_names[0] = ['conv1']
    # W+ index 1
    w_idx_to_module_names[1] = ['to_rgb1']

    # For i from 0 to 7 (8 stages for 1024x1024, if starting at 4x4)
    # The `convs` ModuleList has 16 StyledConv modules (0-15)
    # The `to_rgbs` ModuleList has 8 ToRGB modules (0-7)
    
    for i_stage in range(model_train.n_latent // 2 - 1): # For stages beyond 4x4 and 8x8
        # W+ index for first conv in pair
        w_idx_first_conv = i_stage * 2 + 2
        w_idx_to_module_names.setdefault(w_idx_first_conv, []).append(f'convs.{i_stage*2}')
        
        # W+ index for second conv in pair AND ToRGB
        w_idx_second_conv = i_stage * 2 + 3
        w_idx_to_module_names.setdefault(w_idx_second_conv, []).append(f'convs.{i_stage*2+1}')
        w_idx_to_module_names.setdefault(w_idx_second_conv, []).append(f'to_rgbs.{i_stage}')

    # Now, use `chosen_layer_idx` to get the module names
    actual_modules_to_unfreeze_base_names = set()
    for w_idx in chosen_layer_idx:
        if w_idx in w_idx_to_module_names:
            for name_prefix in w_idx_to_module_names[w_idx]:
                actual_modules_to_unfreeze_base_names.add(name_prefix)

    print(f"  Base modules selected for unfreezing: {list(actual_modules_to_unfreeze_base_names)}")

    # Freeze all parameters by default
    for param in model_train.parameters():
        param.requires_grad = False

    # Unfreeze parameters belonging to the chosen modules by checking parameter names
    for name, param in model_train.named_parameters():
        if any(name.startswith(base_name) for base_name in actual_modules_to_unfreeze_base_names):
            param.requires_grad = True
            # print(f"  Unfreezing parameter: {name}") # Uncomment for verbose debugging
    
    print(f"  Total unfrozen parameters: {sum(1 for p in model_train.parameters() if p.requires_grad)}")


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

def run_on_batch(inputs, model_psp, device): # Add device as argument
    """Runs the pSp model on a batch of data."""
    images, latents = model_psp(inputs.to(device).float(), randomize_noise=False, return_latents=True)
    return images, latents
