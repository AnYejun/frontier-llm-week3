"""
═══════════════════════════════════════════════════════════════
SAE Utilities — Toy Autoencoder & Pre-trained SAE Loading
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class ToyAutoencoder(nn.Module):
    """
    A minimal Sparse Autoencoder for educational purposes.
    
    Architecture:
        Input (d_model) → Encoder (d_sae) → ReLU → Top-K → Decoder (d_model)
    
    The SAE learns to decompose dense representations into sparse,
    interpretable features. The "dimension explosion" (d_sae >> d_model)
    is what allows monosemantic feature learning.
    """
    
    def __init__(self, d_model: int, d_sae: int, top_k: int = 10):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.top_k = top_k
        
        # Encoder: d_model → d_sae
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        
        # Decoder: d_sae → d_model (tied weights possible)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        
        # Initialize with Kaiming
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature space."""
        z = F.relu(self.encoder(x))
        return self._apply_topk(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from sparse features back to model space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sparsify → decode.
        Returns (reconstruction, sparse_features).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def _apply_topk(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Top-K sparsity: keep only the K largest activations."""
        if self.top_k >= z.shape[-1]:
            return z
        
        topk_vals, topk_idx = torch.topk(z, self.top_k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse
    
    def get_active_features(self, x: torch.Tensor) -> List[Tuple[int, float]]:
        """Get list of (feature_index, activation_value) for active features."""
        z = self.encode(x)
        if z.dim() > 1:
            z = z.squeeze(0)
        
        active = []
        for i, val in enumerate(z.tolist()):
            if abs(val) > 1e-6:
                active.append((i, val))
        
        return sorted(active, key=lambda x: abs(x[1]), reverse=True)


def train_toy_sae(
    data: torch.Tensor,
    d_sae: int,
    top_k: int = 10,
    epochs: int = 500,
    lr: float = 1e-3,
    sparsity_weight: float = 1e-3,
    verbose: bool = True
) -> ToyAutoencoder:
    """
    Train a Toy SAE on the given data.
    
    Args:
        data: Training data [N, d_model]
        d_sae: SAE hidden dimension (should be >> d_model)
        top_k: Number of features to keep active
        epochs: Training epochs
        lr: Learning rate
        sparsity_weight: L1 penalty for encouraging sparsity
        verbose: Print training progress
    
    Returns:
        Trained ToyAutoencoder
    """
    d_model = data.shape[1]
    sae = ToyAutoencoder(d_model, d_sae, top_k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    for epoch in range(epochs):
        x_hat, z = sae(data)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, data)
        
        # Sparsity loss (L1 on activations)
        sparse_loss = z.abs().mean()
        
        loss = recon_loss + sparsity_weight * sparse_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Recon: {recon_loss.item():.6f} | "
                  f"Sparse: {sparse_loss.item():.6f} | "
                  f"Total: {loss.item():.6f}")
    
    return sae


def load_pretrained_sae(
    model_name: str = "gpt2",
    layer: int = 6,
    device: str = "cpu"
):
    """
    Load a pre-trained SAE from sae-lens.
    
    Supported models:
        - gpt2: GPT-2 Small (117M) — fast, CPU friendly
        - llama-3: Llama-3-8B — requires GPU / MLX
    
    Returns:
        (sae_model, cfg_dict, sparsity)
    """
    try:
        from sae_lens import SAE
        
        if model_name == "gpt2":
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",
                sae_id=f"blocks.{layer}.hook_resid_pre",
                device=device
            )
            return sae, cfg_dict, sparsity
        elif "llama" in model_name.lower():
            # Note: Llama SAEs are much larger
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release="llama-scope-lxr-8x",
                sae_id=f"layers.{layer}",
                device=device
            )
            return sae, cfg_dict, sparsity
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    except ImportError:
        print("⚠️  sae-lens not installed. Using ToyAutoencoder instead.")
        print("   Install with: pip install sae-lens")
        return None, None, None


def analyze_feature_polysemanticity(
    sae: ToyAutoencoder,
    concepts: List[torch.Tensor],
    concept_names: List[str],
    threshold: float = 0.01
) -> dict:
    """
    Analyze how monosemantic each feature is.
    
    A monosemantic feature activates for exactly one concept.
    A polysemantic feature activates for multiple concepts.
    
    Returns dict with:
        - per_feature: list of (feature_idx, [concept_names_that_activate_it])
        - polysemantic_ratio: fraction of features that are polysemantic
        - monosemantic_ratio: fraction that are monosemantic
    """
    num_features = sae.d_sae
    feature_concepts = {i: [] for i in range(num_features)}
    
    for concept, name in zip(concepts, concept_names):
        active = sae.get_active_features(concept)
        for feat_idx, val in active:
            if abs(val) > threshold:
                feature_concepts[feat_idx].append(name)
    
    poly_count = sum(1 for v in feature_concepts.values() if len(v) > 1)
    mono_count = sum(1 for v in feature_concepts.values() if len(v) == 1)
    dead_count = sum(1 for v in feature_concepts.values() if len(v) == 0)
    
    return {
        'per_feature': [(k, v) for k, v in feature_concepts.items() if v],
        'polysemantic_ratio': poly_count / max(1, num_features),
        'monosemantic_ratio': mono_count / max(1, num_features),
        'dead_ratio': dead_count / max(1, num_features),
        'poly_count': poly_count,
        'mono_count': mono_count,
        'dead_count': dead_count,
    }
