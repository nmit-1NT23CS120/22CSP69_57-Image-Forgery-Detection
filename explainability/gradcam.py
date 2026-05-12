"""
=============================================================
STEP 8b — gradcam.py — Explainability Heatmaps
=============================================================
Generates Grad-CAM heatmaps showing WHERE in the image
the model detected forgery artifacts.
Used by the Flask web app for the results page.
=============================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms


class GradCAM:
    """
    Gradient Class Activation Mapping.

    Shows WHICH regions of the image the model focused on
    when making its forgery detection decision.

    Red regions  = highly suspicious / manipulated areas
    Blue regions = less influential regions
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor : (1 x 3 x 224 x 224) normalized tensor
            class_idx    : class to explain (None = predicted class)

        Returns:
            heatmap : numpy array (224 x 224) values 0-1
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Pool gradients over spatial dimensions
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])

        # Weight activations by gradients
        activations  = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        # Average over channels
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Normalize
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx


def apply_heatmap(original_img_np, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img_np : numpy array (H x W x 3) uint8 RGB
        heatmap         : numpy array (H x W) values 0-1
        alpha           : blending factor (0=original, 1=heatmap)

    Returns:
        overlaid : numpy array (H x W x 3) uint8 RGB
    """
    # Resize heatmap to match image
    h, w = original_img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to colormap (COLORMAP_JET: blue=safe, red=suspicious)
    heatmap_uint8  = np.uint8(255 * heatmap_resized)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color  = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay on original
    overlaid = cv2.addWeighted(
        original_img_np.astype(np.uint8), 1 - alpha,
        heatmap_color,                    alpha,
        0
    )
    return overlaid


def save_heatmap(overlaid_np, save_path):
    """Save heatmap overlay to disk."""
    img = Image.fromarray(overlaid_np.astype(np.uint8))
    img.save(save_path)


# ── Transform for model input ─────────────────────────────────
INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def run_gradcam(model, target_layer, image_path, device,
                save_path=None):
    """
    Full pipeline: load image → run model → generate heatmap.

    Args:
        model        : trained PyTorch model
        target_layer : layer to hook (e.g. model.spatial[-1])
        image_path   : path to input image
        device       : torch device
        save_path    : where to save heatmap (optional)

    Returns:
        result dict with prediction, confidence, trust, heatmap path
    """
    # Load image
    pil_img  = Image.open(image_path).convert('RGB')
    orig_np  = np.array(pil_img.resize((224, 224)))

    # Prepare tensor
    tensor   = INFER_TF(pil_img).unsqueeze(0).to(device)

    # Run Grad-CAM
    gcam     = GradCAM(model, target_layer)
    heatmap, class_idx = gcam.generate(tensor)

    # Get prediction details
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    forged_prob = probs[1].item()
    auth_prob   = probs[0].item()
    predicted   = 'forged' if class_idx == 1 else 'authentic'

    # Trust score
    if forged_prob < 0.3:
        trust = 'LOW RISK'
        trust_color = '#22c55e'
    elif forged_prob < 0.7:
        trust = 'SUSPICIOUS'
        trust_color = '#f59e0b'
    else:
        trust = 'HIGH RISK'
        trust_color = '#ef4444'

    # Generate overlay
    overlay = apply_heatmap(orig_np, heatmap, alpha=0.45)

    # Save if path given
    heatmap_path = None
    if save_path:
        save_heatmap(overlay, save_path)
        heatmap_path = save_path

    return {
        'predicted'    : predicted,
        'forged_prob'  : round(forged_prob * 100, 1),
        'auth_prob'    : round(auth_prob   * 100, 1),
        'trust'        : trust,
        'trust_color'  : trust_color,
        'heatmap_path' : heatmap_path,
        'heatmap_array': overlay,
    }
