import math
from sklearn.decomposition import PCA
import dlib 
from torchvision import transforms 
import re
import os

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
