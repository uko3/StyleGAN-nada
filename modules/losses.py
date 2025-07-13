import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
import math

class CLIPLoss(torch.nn.Module):
    def __init__(self, stylegan_size=1024):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, text_features_normalized): 
        image = self.avg_pool(self.upsample(image))
        image_vector = self.model.encode_image(image)
        image_vector = image_vector / image_vector.norm(dim=-1, keepdim=True)
        similarity = torch.cosine_similarity(image_vector, text_features_normalized, dim=-1) 
        loss = 1 - similarity.mean()
        return loss

class CLIPDirectionalLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPDirectionalLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

    def forward(self, img_frozen, img_styled, text_features_source_norm, text_features_target_norm):
        img_style_np_batch = [(img.detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255 for img in img_styled] 
        img_frozen_np_batch = [(img.detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255 for img in img_frozen]

        image_style_clip_input_batch = torch.cat([self.preprocess(Image.fromarray(img_np.astype(np.uint8))).unsqueeze(0).to(img_styled.device) for img_np in img_style_np_batch])
        image_frozen_clip_input_batch = torch.cat([self.preprocess(Image.fromarray(img_np.astype(np.uint8))).unsqueeze(0).to(img_frozen.device) for img_np in img_frozen_np_batch])

        with torch.no_grad():
            image_features_style = self.model.encode_image(image_style_clip_input_batch)
            image_features_frozen = self.model.encode_image(image_frozen_clip_input_batch)

        image_features_style = image_features_style / image_features_style.norm(dim=-1, keepdim=True)
        image_features_frozen = image_features_frozen / image_features_frozen.norm(dim=-1, keepdim=True)

        enc_images = image_features_style - image_features_frozen
        enc_texts = text_features_target_norm - text_features_source_norm

        cos_sim_val = F.cosine_similarity(enc_images, enc_texts, dim=-1).clamp(-1, 1).mean().item()
        # angle_deg = torch.acos(torch.tensor(cos_sim_val, device=img_styled.device)).item() * 180 / math.pi
        # print(f"Cosine sim: {cos_sim_val:.4f}", f"Angle between directions: {angle_deg:.2f}°") # Для отладки

        loss_clip = 1 - cos_sim_val
        return loss_clip
