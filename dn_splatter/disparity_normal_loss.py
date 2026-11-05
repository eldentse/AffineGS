import torch
import torch.nn.functional as F
import numpy as np
import cv2
from functools import lru_cache
import matplotlib.pyplot as plt
import os

_MAX_DIRECT_GAUSSIAN_KERNEL_SIZE = 129


def _nonzero_sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def _safe_norm_dim0(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(x.pow(2).sum(dim=0, keepdim=True) + eps)


def _compute_tensor_dd_duv(tensor: torch.Tensor) -> torch.Tensor:
    """Compute finite-difference gradients du and dv of a [1, H, W] tensor."""
    assert tensor.ndim == 3, tensor.shape
    dd_du = torch.zeros_like(tensor)
    dd_dv = torch.zeros_like(tensor)

    dd_du[:, :, 1:-1] = tensor[:, :, 2:] - tensor[:, :, :-2]
    dd_dv[:, 1:-1, :] = tensor[:, 2:, :] - tensor[:, :-2, :]

    return torch.cat([dd_du, dd_dv], dim=0) / 2


def _cosine_dudv_loss(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    t1 = tensor1 / _safe_norm_dim0(tensor1, eps=1e-8)
    t2 = tensor2 / _safe_norm_dim0(tensor2, eps=1e-8)
    cos = (t1 * t2).sum(dim=0)
    cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
    return (1 - cos).mean()


def _combine_dn_loss_terms(
    mse_loss: torch.Tensor, cosine_loss: torch.Tensor, dn_loss_type: str
) -> torch.Tensor:
    if dn_loss_type == "mse":
        return mse_loss
    if dn_loss_type == "cosine":
        return cosine_loss
    if dn_loss_type == "both":
        return mse_loss + cosine_loss
    raise ValueError(f"Unsupported dn_loss_type '{dn_loss_type}'. Expected one of: mse, cosine, both.")


@lru_cache(maxsize=None)
def _gaussian_kernel_1d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _opencv_default_gaussian_sigma(kernel_size: int) -> float:
    return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8


def _make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _gaussian_blur_tensor_exact(
    tensor: torch.Tensor, kernel_size: int, sigma: float = None
) -> torch.Tensor:
    if kernel_size <= 1:
        return tensor

    if sigma is None:
        sigma = _opencv_default_gaussian_sigma(kernel_size)
    kernel = _gaussian_kernel_1d(kernel_size, sigma, tensor.device, tensor.dtype)
    kernel_x = kernel.view(1, 1, 1, kernel_size)
    kernel_y = kernel.view(1, 1, kernel_size, 1)

    image = tensor.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    image = F.pad(image, (pad, pad, 0, 0), mode="reflect")
    image = F.conv2d(image, kernel_x)
    image = F.pad(image, (0, 0, pad, pad), mode="reflect")
    image = F.conv2d(image, kernel_y)
    return image.squeeze(0).squeeze(0)


def _large_gaussian_blur_tensor(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Approximate very wide Gaussian blur by blurring a downsampled image.

    A radius-600 full-resolution separable convolution is prohibitively expensive
    here. The result is already extremely low-frequency, so coarse blur plus
    bilinear upsampling preserves the intended smoothing at a fraction of the cost.
    """
    height, width = tensor.shape[-2:]
    scale = max(
        1,
        int(np.ceil(kernel_size / _MAX_DIRECT_GAUSSIAN_KERNEL_SIZE)),
    )
    scale = min(scale, max(1, min(height, width) // 8))

    if scale <= 1:
        return _gaussian_blur_tensor_exact(tensor, kernel_size)

    down_height = max(2, int(round(height / scale)))
    down_width = max(2, int(round(width / scale)))
    downsampled = F.interpolate(
        tensor[None, None],
        size=(down_height, down_width),
        mode="area",
    )

    down_kernel_size = _make_odd(max(3, int(np.ceil(kernel_size / scale))))
    max_kernel_size = 2 * min(down_height, down_width) - 1
    down_kernel_size = min(down_kernel_size, max_kernel_size)
    down_kernel_size = down_kernel_size if down_kernel_size % 2 == 1 else down_kernel_size - 1

    sigma = _opencv_default_gaussian_sigma(kernel_size) / scale
    blurred = _gaussian_blur_tensor_exact(
        downsampled.squeeze(0).squeeze(0),
        down_kernel_size,
        sigma=sigma,
    )
    return F.interpolate(
        blurred[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def _gaussian_blur_tensor(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= _MAX_DIRECT_GAUSSIAN_KERNEL_SIZE:
        return _gaussian_blur_tensor_exact(tensor, kernel_size)
    return _large_gaussian_blur_tensor(tensor, kernel_size)


@lru_cache(maxsize=None)
def _create_grid(width, height, device, dtype=torch.float):
    yy, xx = torch.meshgrid(
        torch.arange(0, height, device=device, dtype=dtype),
        torch.arange(0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0) + 0.5


def _compute_dd_duv(normal, depth, focal_x, focal_y, cx, cy):
    """Compute depth gradients du/dv from normals and camera intrinsics."""
    grid = _create_grid(
        width=depth.shape[2],
        height=depth.shape[1],
        device=normal.device,
        dtype=normal.dtype,
    )

    nz = normal[2:3]
    nz_safe = torch.abs(nz).clamp(min=1e-4) * _nonzero_sign(nz)
    Z = (
        1
        + (normal[0:1] * (grid[0:1] - cx) / nz_safe / focal_x)
        + (normal[1:2] * (grid[1:2] - cy) / nz_safe / focal_y)
    ) * nz_safe

    Z_safe = torch.abs(Z).clamp(min=1e-6) * _nonzero_sign(Z)
    dU_du = (-normal[0:1] * depth) / focal_x / Z_safe
    dU_dv = (-normal[1:2] * depth) / focal_y / Z_safe

    return torch.cat([dU_du, dU_dv], dim=0)


def disparity_normal_loss(
    render_depth: torch.Tensor,
    gt_disparity: torch.Tensor,
    depth_weight: float = 0.1,
    l1_weight: float = 0.4,
    render_normal_weight: float = 0.1,
    render_normals: torch.Tensor = None,
    focal_x: float = None,
    focal_y: float = None,
    cx: float = None,
    cy: float = None,
    guided_blur_radius: int = 800,
    guided_blur_eps: float = 1.0,
    blur_mode: str = "guided",
    dn_loss_type: str = "both",
) -> torch.Tensor:

    render_depth = torch.nan_to_num(render_depth, nan=0.0, posinf=0.0, neginf=0.0).float()
    gt_disparity = torch.nan_to_num(gt_disparity, nan=0.0, posinf=0.0, neginf=0.0).float()

    render_disp = 1.0 / render_depth

    gt_disparity = gt_disparity * (render_disp.max() - render_disp.min()) + render_disp.min()

    d_gt_dudv = _compute_tensor_dd_duv(gt_disparity)
    d_render_dudv = _compute_tensor_dd_duv(render_depth)

    d_gt_norm = torch.sqrt(d_gt_dudv.pow(2).sum(dim=0) + 1e-8)
    d_render_norm = torch.sqrt(d_render_dudv.pow(2).sum(dim=0) + 1e-8)

    gt_safe = gt_disparity

    coefficients = d_gt_norm / (gt_safe.squeeze(0) ** 2 * d_render_norm + 1e-5)

    # Clamp 10% of outliers (bottom 5% and top 5%)
    coef_lower = torch.quantile(coefficients.float(), 0.05)
    coef_upper = torch.quantile(coefficients.float(), 0.95)
    coefficients = coefficients.clamp(min=coef_lower, max=coef_upper)

    beta = 1.0 / gt_safe.squeeze(0) - coefficients * render_depth.squeeze(0)

    if blur_mode == "guided":
        gt_guide = gt_safe[0].detach().cpu().numpy().astype(np.float32)
        coefficients = torch.from_numpy(
            cv2.ximgproc.guidedFilter(gt_guide, coefficients.detach().cpu().numpy().astype(np.float32), radius=guided_blur_radius, eps=guided_blur_eps, dDepth=-1)
        ).to(gt_safe.device)
        beta = torch.from_numpy(
            cv2.ximgproc.guidedFilter(gt_guide, beta.detach().cpu().numpy().astype(np.float32), radius=guided_blur_radius, eps=guided_blur_eps, dDepth=-1)
        ).to(gt_safe.device)
    elif blur_mode == "gaussian":
        gaussian_kernel_size = max(1, int(guided_blur_radius) * 2 + 1)
        coefficients = _gaussian_blur_tensor(coefficients, gaussian_kernel_size)
        beta = _gaussian_blur_tensor(beta, gaussian_kernel_size)
    elif blur_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported blur_mode '{blur_mode}'. Expected one of: guided, gaussian, none.")

    # The guided filter is a local linear fit and can push coefficients to <= 0;
    # keep them in their pre-blur range so 1 / (coefficients * gt) stays finite.
    coefficients = coefficients.clamp(min=coef_lower, max=coef_upper)

    d_gt_dudv_pred =-coefficients * gt_safe.squeeze(0) ** 2 * d_render_dudv

    d_gt_dudv_pred_norm = torch.sqrt(d_gt_dudv_pred.pow(2).sum(dim=0) + 1e-8)
    d_gt_dudv_norm = torch.sqrt(d_gt_dudv.pow(2).sum(dim=0) + 1e-8)

    # Clamp 10% of outliers (bottom 5% and top 5%) 
    lower = torch.quantile(d_gt_dudv_pred_norm.float(), 0.05)
    upper = torch.quantile(d_gt_dudv_pred_norm.float(), 0.95)
    d_gt_dudv_pred_norm = d_gt_dudv_pred_norm.clamp(min=lower, max=upper)

    lower = torch.quantile(d_gt_dudv_norm.float(), 0.05)
    upper = torch.quantile(d_gt_dudv_norm.float(), 0.95)
    d_gt_dudv_norm = d_gt_dudv_norm.clamp(min=lower, max=upper)

    render_depth_reconstructed = (1 - beta * gt_safe.squeeze(0)) / (coefficients * gt_safe.squeeze(0))

    # Align reconstructed depth to render depth using least squares
    with torch.no_grad():
        _fx = render_depth_reconstructed.flatten()
        _fy = render_depth.squeeze(0).flatten()
        _fA = torch.stack([_fx, torch.ones_like(_fx)], dim=1)
        _sol = torch.linalg.solve(_fA.T @ _fA, _fA.T @ _fy)
        _fa, _fb = _sol[0], _sol[1]
    render_depth_reconstructed = _fa * render_depth_reconstructed + _fb

    # L1 loss (in depth space)
    L1_depth_loss = (render_depth_reconstructed - render_depth.squeeze(0)).abs().mean()

    depth_mse_loss = (d_gt_dudv_pred_norm - d_gt_dudv_norm).pow(2).mean()
    depth_cosine_loss = _cosine_dudv_loss(
        tensor1=d_gt_dudv_pred,
        tensor2=d_gt_dudv,
    )
    depth_loss = _combine_dn_loss_terms(depth_mse_loss, depth_cosine_loss, dn_loss_type) * depth_weight

    # Render normals branch
    render_normal_loss = torch.tensor(0.0, device=render_depth.device)
    if render_normals is not None:
        d_render_dudv_from_normal = _compute_dd_duv(
            normal=render_normals,
            depth=render_depth,
            focal_x=focal_x,
            focal_y=focal_y,
            cx=cx,
            cy=cy,
        )
        d_render_dudv_from_normal = torch.nan_to_num(
            d_render_dudv_from_normal, nan=1e-6
        )

        d_gt_dudv_pred2 = -coefficients * gt_safe.squeeze(0) ** 2 * d_render_dudv_from_normal

        d_gt_dudv_pred2_norm = torch.sqrt(
            d_gt_dudv_pred2.pow(2).sum(dim=0) + 1e-8
        )

        # Clamp 10% of outliers (bottom 5% and top 5%)
        lower = torch.quantile(d_gt_dudv_pred2_norm.float(), 0.05)
        upper = torch.quantile(d_gt_dudv_pred2_norm.float(), 0.95)
        d_gt_dudv_pred2_norm = d_gt_dudv_pred2_norm.clamp(min=lower, max=upper)

        render_normal_mse_loss = (d_gt_dudv_pred2_norm - d_gt_dudv_norm).pow(2).mean()
        render_normal_cosine_loss = _cosine_dudv_loss(
            tensor1=d_gt_dudv_pred2,
            tensor2=d_gt_dudv,
        )
        render_normal_loss = (
            _combine_dn_loss_terms(render_normal_mse_loss, render_normal_cosine_loss, dn_loss_type)
            * render_normal_weight
        )

    return (depth_loss, render_normal_loss, L1_depth_loss * l1_weight)
