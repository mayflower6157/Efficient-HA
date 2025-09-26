import torch

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


from PIL import Image
import numpy as np
import torch

def add_diffusion_noise_pil(image: Image.Image, noise_step: int, num_steps: int = 1000, preserve_alpha: bool = True) -> Image.Image:
    """
    对输入 PIL.Image 按给定扩散步 t 加噪，返回 PIL.Image。
    - noise_step: 0 ~ num_steps-1
    - preserve_alpha: 若有 alpha 通道，是否保持 alpha 不变（默认保持）
    """
    # ---- helper: PIL -> torch tensor in [0,1], CxHxW ----
    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.uint8, copy=True)  # 关键：copy=True
        if arr.ndim == 2:               # HxW -> HxWx1
            arr = arr[..., None]
        t = torch.from_numpy(arr).to(torch.float32).div_(255.0)
        t = t.permute(2, 0, 1).contiguous()  # CxHxW
        return t


    # ---- helper: torch tensor [0,1], CxHxW -> PIL ----
    def tensor_to_pil(t: torch.Tensor, mode_hint=None) -> Image.Image:
        t = t.clamp(0, 1).mul(255).round().to(torch.uint8)
        t = t.permute(1, 2, 0).cpu().numpy()   # HxWxC
        if t.shape[2] == 1:
            t = t.squeeze(2)
            return Image.fromarray(t, mode='L')
        if mode_hint in ('RGB', 'RGBA', 'L'):
            return Image.fromarray(t, mode=mode_hint if t.shape[2] in (1,3,4) else None)
        return Image.fromarray(t)

    # 记录原始模式 & 是否分离 alpha
    orig_mode = image.mode
    if preserve_alpha and orig_mode in ('RGBA', 'LA'):
        # 拆出 alpha，噪声只加到颜色通道
        rgba = image.convert('RGBA')
        rgb = rgba.convert('RGB')
        alpha = np.array(rgba.split()[-1], copy=True)  # HxW uint8
        x0 = pil_to_tensor(rgb)                        # 3xHxW
        has_alpha = True
    else:
        # 统一到 RGB 或 L
        if orig_mode not in ('RGB', 'L'):
            work = image.convert('RGB')
        else:
            work = image
        x0 = pil_to_tensor(work)                       # CxHxW
        has_alpha = False

    # 归一化步数
    t = int(max(0, min(num_steps - 1, int(noise_step))))

    # ---- 同你原来代码的 beta/alpha 调度 ----
    betas = torch.linspace(-6, 6, num_steps).sigmoid() * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    a = torch.sqrt(alphas_bar[t]).to(x0.dtype)
    b = torch.sqrt(1.0 - alphas_bar[t]).to(x0.dtype)

    # q(x_t | x_0) 采样
    noise = torch.randn_like(x0)
    xt = a * x0 + b * noise

    # 回到 PIL
    out = tensor_to_pil(xt, mode_hint='RGB' if x0.shape[0] == 3 else 'L')

    if has_alpha:
        out = out.convert('RGBA')
        out_np = np.array(out, copy=True)
        out_np[..., 3] = alpha  # 还原 alpha
        out = Image.fromarray(out_np, mode='RGBA')

    # 尽量还原到原模式（若可行）
    if orig_mode == 'L' and out.mode != 'L':
        out = out.convert('L')
    elif orig_mode == 'RGB' and out.mode != 'RGB':
        out = out.convert('RGB')
    elif orig_mode == 'RGBA' and out.mode != 'RGBA':
        out = out.convert('RGBA')

    return out

# 用法示例
# img = Image.open("in.png")
# noisy = add_diffusion_noise_pil(img, noise_step=500)
# noisy.save("noisy.png")
