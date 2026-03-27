"""
═══════════════════════════════════════════════════════════════
Steering Utilities — Contrastive Vectors & Norm-Preserving Ops
Based on REINS framework patterns.
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SteeringResult:
    """Result of a steering operation."""
    original_norm: float
    steered_norm: float
    norm_deviation: float  # percentage
    cosine_shift: float    # how much direction changed
    intervention_applied: bool


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP).
    Interpolates between v0 and v1 on the unit hypersphere.
    
    Args:
        v0: Start vector
        v1: End vector (target direction)
        t: Interpolation factor [0, 1]
    
    Returns:
        Interpolated vector with preserved norm of v0
    """
    original_norm = torch.norm(v0)
    v0_unit = v0 / (torch.norm(v0) + 1e-8)
    v1_unit = v1 / (torch.norm(v1) + 1e-8)
    
    dot = torch.clamp(torch.dot(v0_unit.flatten(), v1_unit.flatten()), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    
    if sin_omega.abs() < 1e-6:
        # Vectors are nearly parallel, use linear interpolation
        result = (1.0 - t) * v0 + t * v1
    else:
        s0 = torch.sin((1.0 - t) * omega) / sin_omega
        s1 = torch.sin(t * omega) / sin_omega
        result = s0 * v0 + s1 * v1
    
    # Preserve original norm
    return result * (original_norm / (torch.norm(result) + 1e-8))


def compute_contrastive_vector(
    model,
    tokenizer,
    positive_texts: List[str],
    negative_texts: List[str],
    layer: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute a contrastive steering vector from positive/negative anchor texts.
    
    v_steering = centroid(positive_activations) - centroid(negative_activations)
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        positive_texts: Texts representing the desired behavior
        negative_texts: Texts representing the opposite behavior
        layer: Target layer index
        device: Compute device
    
    Returns:
        Steering vector (d_model dimensional)
    """
    def get_activations(texts: List[str]) -> torch.Tensor:
        all_acts = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # Get last token's hidden state at target layer
            hidden = outputs.hidden_states[layer]
            last_token = hidden[0, -1, :]  # [d_model]
            all_acts.append(last_token)
        return torch.stack(all_acts)
    
    pos_acts = get_activations(positive_texts)
    neg_acts = get_activations(negative_texts)
    
    pos_centroid = pos_acts.mean(dim=0)
    neg_centroid = neg_acts.mean(dim=0)
    
    steering_vector = pos_centroid - neg_centroid
    return steering_vector


def apply_norm_preserving_steering(
    h: torch.Tensor,
    v_steering: torch.Tensor,
    strength: float = 0.1,
    threshold: float = 0.7,
    method: str = 'slerp'
) -> Tuple[torch.Tensor, SteeringResult]:
    """
    Anyed's Norm-Preserving Steering Algorithm.
    
    1. Orthogonalize steering vector w.r.t. hidden state
    2. Check deviation threshold
    3. Apply rotation preserving ||h||
    
    Args:
        h: Hidden state vector [d_model]
        v_steering: Steering direction [d_model]
        strength: Steering intensity λ (0.0 = no change, 1.0 = full rotation)
        threshold: Deviation threshold for intervention
        method: 'slerp' or 'additive'
    
    Returns:
        Tuple of (steered_hidden_state, steering_result)
    """
    original_norm = torch.norm(h).item()
    h_unit = h / (torch.norm(h) + 1e-8)
    v_unit = v_steering / (torch.norm(v_steering) + 1e-8)
    
    # Cosine similarity between current state and steering direction
    sim = cosine_similarity(h, v_steering)
    
    if sim >= threshold:
        # Already aligned, no intervention needed
        return h, SteeringResult(
            original_norm=original_norm,
            steered_norm=original_norm,
            norm_deviation=0.0,
            cosine_shift=0.0,
            intervention_applied=False
        )
    
    if method == 'slerp':
        # Spherical interpolation toward steering direction
        target = v_unit * torch.norm(h)
        h_steered = slerp(h, target, strength)
    else:
        # Additive with orthogonal projection
        radial_proj = torch.dot(v_steering.flatten(), h_unit.flatten())
        v_perp = v_steering - radial_proj * h_unit
        v_perp_unit = v_perp / (torch.norm(v_perp) + 1e-8)
        
        alpha = (threshold - sim) * strength
        h_steered = h + alpha * torch.norm(h) * v_perp_unit
        
        # Safety Lock: Force Renormalization
        h_steered = h_steered * (torch.norm(h) / (torch.norm(h_steered) + 1e-8))
    
    steered_norm = torch.norm(h_steered).item()
    norm_dev = abs(steered_norm - original_norm) / (original_norm + 1e-8) * 100
    cos_shift = cosine_similarity(h, h_steered)
    
    return h_steered, SteeringResult(
        original_norm=original_norm,
        steered_norm=steered_norm,
        norm_deviation=norm_dev,
        cosine_shift=1.0 - cos_shift,
        intervention_applied=True
    )


def create_steering_hook(
    steering_vector: torch.Tensor,
    strength: float = 0.1,
    threshold: float = 0.7,
    method: str = 'slerp'
):
    """
    Create a forward hook function for steering injection.
    
    Usage:
        hook = create_steering_hook(v_steering, strength=0.05)
        handle = model.transformer.h[layer].register_forward_hook(hook)
        # ... generate ...
        handle.remove()
    """
    stats = {'interventions': 0, 'total': 0, 'norm_devs': []}
    
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        batch_size, seq_len, d_model = hidden.shape
        
        modified = hidden.clone()
        for s in range(seq_len):
            vec = hidden[0, s, :]
            steered, result = apply_norm_preserving_steering(
                vec, steering_vector, strength, threshold, method
            )
            modified[0, s, :] = steered
            stats['total'] += 1
            if result.intervention_applied:
                stats['interventions'] += 1
                stats['norm_devs'].append(result.norm_deviation)
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    
    return hook_fn, stats


def compute_perplexity(model, tokenizer, text: str, device: str = 'cpu') -> float:
    """
    Compute perplexity of text under the model.
    Used to measure the "Alignment Tax" — how much steering degrades fluency.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    
    return torch.exp(loss).item()
