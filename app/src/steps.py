import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def safe_collate(batch):
    """Fixes negative stride numpy arrays by copying before tensor conversion."""
    imgs, kps, names = zip(*batch)
    imgs = [torch.tensor(np.ascontiguousarray(img)) for img in imgs]
    kps = [torch.tensor(np.ascontiguousarray(kp)) for kp in kps]
    return torch.stack(imgs), torch.stack(kps), names

# Example DataLoader usage:
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=safe_collate)


# -----------------------------
# Training loop
# -----------------------------
def train_one_epoch(model, dataloader, postprocess, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Train]")

    for step, (imgs, keypoints, _) in pbar:
        imgs = imgs.to(device, dtype=torch.float32)
        keypoints = keypoints.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(imgs)  # (B, K, H, W)
        preds = postprocess(outputs)  # (B, K, 2)

        # scale GT to output resolution
        H_out, W_out = outputs.shape[2:]
        H_in, W_in = imgs.shape[2:]
        scale_x, scale_y = W_out / W_in, H_out / H_in
        target_scaled = keypoints.clone()
        target_scaled[..., 0] *= scale_x
        target_scaled[..., 1] *= scale_y

        loss = criterion(preds, target_scaled)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)


# -----------------------------
# Validation loop
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader, postprocess, criterion, device, epoch, args=None):
    model.eval()
    total_loss = 0.0
    for imgs, keypoints, _ in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
        imgs = imgs.to(device, dtype=torch.float32)
        keypoints = keypoints.to(device, dtype=torch.float32)

        outputs = model(imgs)
        preds = postprocess(outputs)

        H_out, W_out = outputs.shape[2:]
        H_in, W_in = imgs.shape[2:]
        scale_x, scale_y = W_out / W_in, H_out / H_in
        target_scaled = keypoints.clone()
        target_scaled[..., 0] *= scale_x
        target_scaled[..., 1] *= scale_y

        loss = criterion(preds, target_scaled)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# -----------------------------
# Full training wrapper
# -----------------------------
def train_model(model, train_loader, val_loader, postprocess, criterion, optimizer, scheduler, args, device="cuda"):
    """
    Training function with learning rate scheduler
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        args: Training arguments dictionary
    """
    best_val_loss = float('inf')
    
    for epoch in range(args['num_epochs']):
        # Training phase
        train_loss = train_one_epoch(model, train_loader, postprocess, optimizer, criterion, device, epoch)
        
        # Validation phase
        # if (epoch + 1) % args['val_intervals'] == 0 or epoch == args['num_epochs'] - 1:
        #     val_loss = evaluate(model, val_loader, criterion)
            
        #     # Save best model
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         torch.save(model.state_dict(), 'best_model.pth')
        #         print(f"Saved best model at epoch {epoch+1} with val_loss: {val_loss:.6f}")
        
        # Step the scheduler (update learning rate)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{args['num_epochs']}] "
              f"Train Loss: {train_loss:.6f} | "
              f"LR: {current_lr:.6f} -> {new_lr:.6f}")
    
    return model