#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
Lab 2: The Concept Surgeon — 개념 수술실
═══════════════════════════════════════════════════════════════

GPT-2 Small을 사용하여 모델 내부의 개념 벡터를 추출하고
조작하는 실험입니다.

Usage:
    python lab2_concept_surgeon.py --demo
    python lab2_concept_surgeon.py --concept sarcasm --strength 1.5 --layer 8
"""

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.steering import (
    compute_contrastive_vector,
    apply_norm_preserving_steering,
    create_steering_hook,
    compute_perplexity,
    cosine_similarity,
    slerp,
    SteeringResult,
)


# ═══════════════════════════════════════════════
# Concept Definitions (Contrastive Anchors)
# ═══════════════════════════════════════════════

CONCEPTS = {
    "sarcasm": {
        "name": "😏 냉소적 말투 (Sarcasm)",
        "positive": [
            "Oh sure, that's a brilliant idea, nobody has ever thought of that before.",
            "What a surprise, another meeting that could have been an email.",
            "Wow, you really outdid yourself with that groundbreaking observation.",
            "Oh please, do enlighten us with your infinite wisdom.",
            "Yes, because that worked so well the last hundred times.",
        ],
        "negative": [
            "That's a wonderful idea! I really appreciate your thoughtful suggestion.",
            "Thank you for sharing your perspective, it's truly valuable.",
            "I'm genuinely impressed by your creativity and insight.",
            "Your contribution makes such a positive difference to the team.",
            "I feel grateful for your kindness and support.",
        ],
    },
    "formal": {
        "name": "🎩 격식체 (Formality)",
        "positive": [
            "I wish to formally express my gratitude for your esteemed consideration.",
            "The aforementioned proposal merits careful deliberation by the committee.",
            "We respectfully submit this document for your review and approval.",
            "It is with great honor that I present the following analysis.",
            "Pursuant to our previous correspondence, I wish to elaborate further.",
        ],
        "negative": [
            "Hey! Thanks a lot, really appreciate it!",
            "So basically what I'm saying is, let's just go for it!",
            "Yo, check this out - pretty cool stuff right?",
            "Lol yeah that makes sense, let's do it!",
            "Nah, I don't think so. Let's try something else!",
        ],
    },
    "creative": {
        "name": "🎨 창의성 (Creativity)",
        "positive": [
            "The sunset painted the sky in hues of molten gold and liquid amethyst, as if the heavens were weeping tears of beauty.",
            "Imagine a world where thoughts bloom like flowers in a garden of consciousness.",
            "The old house whispered secrets through its creaking bones, each groan a forgotten lullaby.",
            "Time folded like origami, each crease revealing a universe of possibility.",
            "She danced with shadows, her movements tracing constellations on the floor.",
        ],
        "negative": [
            "The temperature was 25 degrees Celsius. The humidity was 60%.",
            "The report shows a 3.2% increase in quarterly revenue.",
            "Step 1: Open the file. Step 2: Read the contents. Step 3: Close the file.",
            "The chemical formula for water is H2O. It freezes at 0 degrees.",
            "The population of Tokyo is approximately 13.96 million people.",
        ],
    },
    "honesty": {
        "name": "⚖️ 정직함 (Honesty)",
        "positive": [
            "I have to be honest with you - this isn't going to work as planned.",
            "Let me give you a straightforward answer without sugarcoating it.",
            "The truth is, there are significant risks we need to acknowledge.",
            "I'd rather tell you the uncomfortable truth than a comfortable lie.",
            "Being transparent about the downsides is more important than sounding optimistic.",
        ],
        "negative": [
            "Everything is absolutely perfect and there are zero issues at all!",
            "Don't worry about that, it's totally fine, nothing to see here.",
            "I'm sure it will all work out wonderfully, no need to think about problems.",
            "Let's just focus on the positives and not mention any challenges.",
            "That's a great plan with absolutely no drawbacks whatsoever!",
        ],
    },
    "code_quality": {
        "name": "💻 코드 효율성 (Code Efficiency)",
        "positive": [
            "Use list comprehension instead of explicit loops for better performance.",
            "Memoize expensive function calls with @lru_cache to avoid recomputation.",
            "Prefer generators over lists for large datasets to reduce memory usage.",
            "Use numpy vectorized operations instead of Python loops for numerical work.",
            "Implement async/await for I/O-bound operations to improve throughput.",
        ],
        "negative": [
            "Here's a simple example that's easy to understand for beginners.",
            "Let me walk you through this step by step with lots of comments.",
            "We'll use basic for loops so everyone can follow along easily.",
            "This code prioritizes readability over performance for learning purposes.",
            "Let me explain each line in detail so the logic is crystal clear.",
        ],
    },
}


# ═══════════════════════════════════════════════
# Experiment Functions
# ═══════════════════════════════════════════════

def load_model(device='cpu'):
    """Load GPT-2 Small model and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print("📦 Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   ✅ Model loaded on {device}")
    print(f"   📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🧠 Layers: {model.config.n_layer}, d_model: {model.config.n_embd}")
    return model, tokenizer


def extract_steering_vector(model, tokenizer, concept_key, layer, device='cpu'):
    """Extract a contrastive steering vector for the given concept."""
    concept = CONCEPTS[concept_key]
    print(f"\n🔬 Extracting steering vector for: {concept['name']}")
    print(f"   📍 Target layer: {layer}")
    
    v = compute_contrastive_vector(
        model, tokenizer,
        concept['positive'], concept['negative'],
        layer=layer, device=device
    )
    
    print(f"   📐 Vector norm: {torch.norm(v).item():.4f}")
    print(f"   📏 Dimensions: {v.shape}")
    
    return v


def generate_with_steering(model, tokenizer, prompt, steering_vector, layer, strength, device='cpu', max_tokens=100):
    """Generate text with steering vector applied."""
    hook_fn, stats = create_steering_hook(
        steering_vector,
        strength=abs(strength),
        threshold=0.5 if strength > 0 else 0.3,
        method='slerp'
    )
    
    # If negative strength, flip the vector
    if strength < 0:
        hook_fn_neg, stats = create_steering_hook(
            -steering_vector,
            strength=abs(strength),
            threshold=0.3,
            method='slerp'
        )
        hook_fn = hook_fn_neg
    
    # Register hook
    handle = model.transformer.h[layer].register_forward_hook(hook_fn)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Remove hook
    handle.remove()
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, stats


def run_before_after_comparison(model, tokenizer, concept_key, prompt, layer, strengths, device='cpu'):
    """Run before/after comparison at multiple steering strengths."""
    v = extract_steering_vector(model, tokenizer, concept_key, layer, device)
    
    print(f"\n{'='*60}")
    print(f"📝 Prompt: \"{prompt}\"")
    print(f"{'='*60}")
    
    results = []
    
    for strength in strengths:
        print(f"\n{'─'*40}")
        print(f"🎛️  λ = {strength:+.1f}")
        print(f"{'─'*40}")
        
        if strength == 0:
            # Baseline
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            stats = {'interventions': 0, 'total': 0, 'norm_devs': []}
        else:
            text, stats = generate_with_steering(
                model, tokenizer, prompt, v, layer, strength, device, max_tokens=80
            )
        
        intervention_rate = (stats['interventions'] / max(1, stats['total'])) * 100
        avg_norm_dev = np.mean(stats['norm_devs']) if stats['norm_devs'] else 0
        
        print(f"📊 Intervention rate: {intervention_rate:.1f}%")
        print(f"📐 Avg norm deviation: {avg_norm_dev:.6f}%")
        print(f"💬 Response:\n{text}\n")
        
        results.append({
            'strength': strength,
            'text': text,
            'intervention_rate': intervention_rate,
            'norm_deviation': avg_norm_dev,
        })
    
    return results


def run_conflict_scenario(model, tokenizer, layer, device='cpu'):
    """
    Conflict Resolver: Test what happens when honesty vector
    conflicts with a "white lie" prompt.
    """
    print("\n" + "="*60)
    print("⚔️  CONFLICT RESOLVER: 정직함 vs 선의의 거짓말")
    print("="*60)
    
    # Extract honesty vector
    v_honesty = extract_steering_vector(model, tokenizer, 'honesty', layer, device)
    
    conflict_prompt = "Your friend asks you 'Do I look good in this outfit?' but they don't look great. What do you say?"
    
    print(f"\n📝 Conflict Prompt: \"{conflict_prompt}\"")
    
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    for strength in strengths:
        print(f"\n{'─'*40}")
        print(f"⚖️  Honesty λ = {strength:+.1f}")
        
        if strength == 0:
            inputs = tokenizer(conflict_prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=60,
                    do_sample=True, temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            text, stats = generate_with_steering(
                model, tokenizer, conflict_prompt, v_honesty,
                layer, strength, device, max_tokens=60
            )
        
        # Measure alignment with both vectors
        inputs = tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[layer][0, -1, :]
        
        cos_honesty = cosine_similarity(h, v_honesty)
        cos_original = cosine_similarity(h, -v_honesty)
        
        print(f"   cos(h, v_honesty) = {cos_honesty:.4f}")
        print(f"   cos(h, v_diplomacy) = {cos_original:.4f}")
        print(f"   Drift (honesty - diplomacy) = {(cos_honesty - cos_original):+.4f}")
        print(f"   💬 {text[:200]}")


def run_sae_analysis(model, tokenizer, concept_key, layer, device='cpu'):
    """
    Analyze concept representation using a Toy SAE.
    Shows how sparse features decompose steering vectors.
    """
    from utils.sae_utils import ToyAutoencoder, train_toy_sae, analyze_feature_polysemanticity
    
    print("\n" + "="*60)
    print("🔬 SAE Feature Analysis")
    print("="*60)
    
    # Collect activations for multiple concepts
    concept_keys = list(CONCEPTS.keys())[:3]
    all_acts = []
    concept_names = []
    
    for ck in concept_keys:
        c = CONCEPTS[ck]
        for text in c['positive'][:2] + c['negative'][:2]:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[layer][0, -1, :]
            all_acts.append(h)
        concept_names.append(c['name'])
    
    data = torch.stack(all_acts)
    d_model = data.shape[1]
    
    # Train SAE with different d_sae values
    for d_sae in [128, 512, 2048]:
        top_k = max(5, d_sae // 20)
        print(f"\n📐 Training SAE: d_model={d_model} → d_sae={d_sae} (top-{top_k})")
        
        sae = train_toy_sae(data, d_sae=d_sae, top_k=top_k, epochs=300, verbose=False)
        
        # Reconstruct and measure error
        with torch.no_grad():
            x_hat, z = sae(data)
            recon_error = torch.mean((data - x_hat) ** 2).item()
        
        # Analyze polysemanticity
        concept_acts = []
        for ck in concept_keys:
            c = CONCEPTS[ck]
            text = c['positive'][0]
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            concept_acts.append(outputs.hidden_states[layer][0, -1, :])
        
        analysis = analyze_feature_polysemanticity(sae, concept_acts, concept_names)
        
        print(f"   📊 Reconstruction MSE: {recon_error:.6f}")
        print(f"   🟩 Monosemantic features: {analysis['mono_count']} ({analysis['monosemantic_ratio']*100:.1f}%)")
        print(f"   🟥 Polysemantic features: {analysis['poly_count']} ({analysis['polysemantic_ratio']*100:.1f}%)")
        print(f"   ⬛ Dead features: {analysis['dead_count']} ({analysis['dead_ratio']*100:.1f}%)")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Lab 2: The Concept Surgeon")
    parser.add_argument('--demo', action='store_true', help='Run the full demo')
    parser.add_argument('--concept', type=str, default='sarcasm',
                       choices=list(CONCEPTS.keys()),
                       help='Concept to steer')
    parser.add_argument('--strength', type=float, default=1.0, help='Steering strength λ')
    parser.add_argument('--layer', type=int, default=8, help='Target layer (0-11 for GPT-2)')
    parser.add_argument('--prompt', type=str, default="Tell me about the weather today.",
                       help='Input prompt')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--sae', action='store_true', help='Run SAE analysis')
    parser.add_argument('--conflict', action='store_true', help='Run conflict scenario')
    args = parser.parse_args()
    
    print("🧬 ═══════════════════════════════════════════")
    print("   Lab 2: The Concept Surgeon (개념 수술실)")
    print("   Frontier LLM Week 3")
    print("═══════════════════════════════════════════════")
    
    model, tokenizer = load_model(args.device)
    
    if args.demo:
        print("\n🎯 Running full demo...\n")
        
        # 1. Before/After comparison for each concept
        for concept_key in ['sarcasm', 'formal', 'creative']:
            run_before_after_comparison(
                model, tokenizer, concept_key,
                args.prompt, args.layer,
                strengths=[-1.0, 0.0, 1.0, 2.0],
                device=args.device
            )
        
        # 2. Conflict scenario
        run_conflict_scenario(model, tokenizer, args.layer, args.device)
        
        # 3. SAE analysis
        run_sae_analysis(model, tokenizer, args.concept, args.layer, args.device)
        
    elif args.conflict:
        run_conflict_scenario(model, tokenizer, args.layer, args.device)
        
    elif args.sae:
        run_sae_analysis(model, tokenizer, args.concept, args.layer, args.device)
        
    else:
        # Single concept experiment
        run_before_after_comparison(
            model, tokenizer, args.concept,
            args.prompt, args.layer,
            strengths=[0.0, args.strength],
            device=args.device
        )
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
