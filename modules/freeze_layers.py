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

