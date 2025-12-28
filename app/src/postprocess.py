import torch
import torch.nn.functional as F


# -----------------------------
# Differentiable postprocess: Soft-Argmax
# -----------------------------
def court_postprocess(output):
    """
    Args:
        output: (B, K, H, W)
    Returns:
        coords: (B, K, 2) in pixel coordinates
    """
    B, K, H, W = output.shape

    # Flatten heatmaps
    heatmaps = output.view(B, K, -1)
    heatmaps = F.softmax(heatmaps, dim=-1)  # normalized probabilities

    # Create coordinate grids
    xs = torch.linspace(0, W - 1, W, device=output.device)
    ys = torch.linspace(0, H - 1, H, device=output.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    exp_x = torch.sum(heatmaps * grid_x, dim=-1)
    exp_y = torch.sum(heatmaps * grid_y, dim=-1)

    coords = torch.stack([exp_x, exp_y], dim=2)
    return coords

def ball_postprocess(output):
    """
    Soft-argmax for heatmap regression.

    Args:
        output: Tensor of shape (B, 4, H, W)
                4 heatmaps correspond to x1, y1, x2, y2

    Returns:
        coords: Tensor of shape (B, 4)
                (x1, y1, x2, y2) in pixel coordinates
    """

    B, K, H, W = output.shape
    assert K == 4, f"Expected 4 channels for xyxy, got {K}"

    # (B, K, H*W)
    heatmaps = output.reshape(B, K, -1)

    # Softmax on each heatmap separately
    heatmaps = F.softmax(heatmaps, dim=-1)  # (B, K, H*W)

    # Create coordinate grid
    ys = torch.linspace(0, H - 1, H, device=output.device)
    xs = torch.linspace(0, W - 1, W, device=output.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Flatten grids → (H*W,)
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    # Expected values
    # (B,K, H*W) * (H*W,) → (B,K)
    exp_x = torch.sum(heatmaps * grid_x, dim=-1)
    exp_y = torch.sum(heatmaps * grid_y, dim=-1)

    # Combine in correct xyxy order:
    # x1 = exp_x[:,0], y1 = exp_y[:,1], x2 = exp_x[:,2], y2 = exp_y[:,3]
    coords = torch.stack([
        exp_x[:, 0],  # x1
        exp_y[:, 1],  # y1
        exp_x[:, 2],  # x2
        exp_y[:, 3],  # y2
    ], dim=1)

    return coords
