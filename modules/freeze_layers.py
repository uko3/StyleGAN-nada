import math
from sklearn.decomposition import PCA
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


def freeze_layers(model_train, freeze_mapping=True, freeze_initial_blocks=2, freeze_torgb=True):
    for name, param in model_train.named_parameters():
        param.requires_grad = True  

        if freeze_mapping and name.startswith("style."):
            param.requires_grad = False

        if freeze_initial_blocks >= 1:
            if name.startswith("conv1.") or name.startswith("to_rgb1."):
                param.requires_grad = False

        for i in range(freeze_initial_blocks - 1):
            if name.startswith(f"convs.{i}.") or name.startswith(f"to_rgbs.{i}."):
                param.requires_grad = False

        if freeze_torgb:
            if name.startswith("to_rgb1.") or name.startswith("to_rgbs."):
                param.requires_grad = False


def freeze_layers_adaptive_fine_grained(model_train, model_frozen, text, freeze_conv_weights=True, freeze_to_rgb_weights=True, k = 5):
    batch_size = 2
    latent_dim = 512
    latent_z = torch.randn(batch_size, latent_dim, device=device)
    with torch.no_grad():
        latent_w = model_frozen.style(latent_z)

    latent_w_plus = latent_w.unsqueeze(1).repeat(1, model_frozen.n_latent, 1)

    latent_tensor = torch.Tensor(latent_w_plus.cpu().detach().numpy()).to(device)
    latent_tensor.requires_grad = True

    fl_optimizer = torch.optim.Adam([latent_tensor], lr=0.01) 

    auto_layer_iters = 3

    for _ in range(auto_layer_iters):
        fl_optimizer.zero_grad()
        generated_img_fl, _ = model_train([latent_tensor], input_is_latent=True)
        loss_fl = clip_loss(generated_img_fl, text)
        loss_fl.backward()
        fl_optimizer.step()

    involved_layers = torch.abs(latent_tensor - latent_w_plus).mean(dim=-1).mean(dim=0)
    used_layers = torch.topk(involved_layers, k)[1].cpu().numpy()

    all_children = list(model_train.children())
    potential_layers = [
        all_children[2], 
        all_children[3]   
    ]
    potential_layers.extend(list(all_children[4])) 

    used_modules = [potential_layers[idx] for idx in used_layers]

    named_modules = list(model_train.named_modules())
    used_layer_names = []
    for name, module in named_modules:
        if module in used_modules:
            used_layer_names.append(name)

    for param in model_train.parameters():
        param.requires_grad = False

    for name, param in model_train.named_parameters():
        if any(name.startswith(used_name) for used_name in used_layer_names):
            is_styled_conv = any(used_name.startswith('conv') for used_name in used_layer_names if name.startswith(used_name))
            is_to_rgb = any(used_name.startswith('to_rgb') for used_name in used_layer_names if name.startswith(used_name))

            if is_styled_conv:
                if "modulation" in name or "noise" in name:
                    param.requires_grad = True
                elif "weight" in name and "conv" in name and freeze_conv_weights == False: 
                    param.requires_grad = True
                elif "bias" in name and "conv" in name and freeze_conv_weights == False: 
                    param.requires_grad = True

            elif is_to_rgb:
                if "modulation" in name:
                    param.requires_grad = True
                elif "weight" in name and "conv" in name and freeze_to_rgb_weights == False: 
                    param.requires_grad = True
                elif "bias" in name and "conv" in name and freeze_to_rgb_weights == False: 
                    param.requires_grad = True


def freeze_layers_adaptive_fine_tune(model_train, model_frozen, text, epochs=3, k=6):
    generator_observation = copy.deepcopy(model_train)
    generator_observation.to(device)
    generator_observation.train()

    all_children = list(generator_observation.children())
    potential_layers = all_children[2:4] + list(all_children[4])  

    named_modules = dict(generator_observation.named_modules())
    used_layer_names = [name for name, module in named_modules.items() if module in potential_layers]

    optimizer = torch.optim.Adam(generator_observation.parameters(), lr=0.01)
    val_par = {}

    batch_size = 2
    latent_dim = 512
    latent_z = torch.randn(batch_size, latent_dim, device=device)
    with torch.no_grad():
        latent_w = model_frozen.style(latent_z)

    for _ in range(epochs):
        optimizer.zero_grad()
        generated_img, _ = generator_observation([latent_w], input_is_latent=True, randomize_noise = False)

        loss = clip_loss(generated_img, text)
        loss.backward()
        optimizer.step()

        for name, param in generator_observation.named_parameters():
            if param.requires_grad and any(name.startswith(prefix) for prefix in used_layer_names):
                val_par.setdefault(name, []).append(torch.norm(param.data).item())

    diffs = {
        name: values[-1] - values[0]
        for name, values in val_par.items()
        if len(values) >= 2
    }

    categories = {
        'conv.weight': [],
        'conv.modulation.weight': [],
        'conv.modulation.bias': [],
        'noise.weight': [],
        'modulation.bias': [],
        'bias': []
    }

    for name, diff in diffs.items():
        if re.search(r'\.conv\.weight$', name):
            categories['conv.weight'].append((name, diff))
        elif re.search(r'\.conv\.modulation\.weight$', name):
            categories['conv.modulation.weight'].append((name, diff))
        elif re.search(r'\.conv\.modulation\.bias$', name):
            categories['conv.modulation.bias'].append((name, diff))
        elif re.search(r'\.noise\.weight$', name):
            categories['noise.weight'].append((name, diff))
        elif re.search(r'\.modulation\.bias$', name):
            categories['modulation.bias'].append((name, diff))
        elif re.search(r'\.bias$', name):
            categories['bias'].append((name, diff))

    selected_params = set()
    for cat, items in categories.items():
        topk = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:k]
        selected_params.update(name for name, _ in topk)

    for param in model_train.parameters():
        param.requires_grad = False

    for name, param in model_train.named_parameters():
        if name in selected_params:
            param.requires_grad = True

                
