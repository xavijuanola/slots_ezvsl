#!/usr/bin/env python
"""
Test script para verificar la lógica de permutación de slots (STE sort).
Prueba el código de train.py líneas 396-452.
"""

import torch
import torch.nn.functional as F
import numpy as np


def test_slot_permutation_logic():
    """
    Prueba la lógica de permutación STE usando el mismo código que train.py
    """
    print("=" * 70)
    print("Prueba de Lógica de Permutación de Slots (STE Sort)")
    print("=" * 70)
    
    # Configuración: batch_size=2, num_slots=2, slot_dim=512
    B, N, C = 2, 2, 512
    
    # Crear datos mock de slot outputs
    # Simulamos que aud_slot_out e img_slot_out vienen del modelo
    aud_slot_out = {
        'slots': torch.randn(B, N, C),
        'q': torch.randn(B, N, C),
        'intra_attn': torch.randn(B, N, 49),  # Para imágenes sería 7x7=49
    }
    
    img_slot_out = {
        'slots': torch.randn(B, N, C),
        'q': torch.randn(B, N, C),
        'k': torch.randn(B, 49, C),  # Para imágenes sería 7x7=49
        'intra_attn': torch.randn(B, N, 49),
    }
    
    # Crear escenario controlado donde sabemos qué slot debería ser "target"
    # Batch 0: slot 0 de audio es más similar a slot 0 de imagen
    # Batch 1: slot 1 de audio es más similar a slot 1 de imagen
    torch.manual_seed(42)
    
    # Forzar similitud conocida
    base_vec = torch.randn(C)
    aud_slot_out['slots'][0, 0] = base_vec.clone()
    img_slot_out['slots'][0, 0] = base_vec.clone() + 0.1 * torch.randn(C)
    aud_slot_out['slots'][0, 1] = base_vec.clone() + 2.0 * torch.randn(C)
    img_slot_out['slots'][0, 1] = base_vec.clone() + 2.5 * torch.randn(C)
    
    base_vec2 = torch.randn(C)
    aud_slot_out['slots'][1, 1] = base_vec2.clone()
    img_slot_out['slots'][1, 1] = base_vec2.clone() + 0.1 * torch.randn(C)
    aud_slot_out['slots'][1, 0] = base_vec2.clone() + 2.0 * torch.randn(C)
    img_slot_out['slots'][1, 0] = base_vec2.clone() + 2.5 * torch.randn(C)
    
    print("\n1. Slots originales (antes de permutación):")
    print(f"   Audio slots shape: {aud_slot_out['slots'].shape}")
    print(f"   Image slots shape: {img_slot_out['slots'].shape}")
    
    # === CÓDIGO REAL DE train.py ===
    # Normalize per-slot vectors on channel dim
    aud_slots = F.normalize(aud_slot_out['slots'], dim=2)
    img_slots = F.normalize(img_slot_out['slots'], dim=2)
    
    # Pairwise similarity S[b, n_a, n_i]
    similarity_slots = torch.einsum('bnc,bmc->bnm', aud_slots, img_slots)
    
    print("\n2. Matriz de similitud pairwise:")
    print(f"   Shape: {similarity_slots.shape} (batch, num_audio_slots, num_img_slots)")
    print(f"   Batch 0:")
    print(f"     Audio slot 0 vs Image slot 0: {similarity_slots[0, 0, 0]:.4f}")
    print(f"     Audio slot 0 vs Image slot 1: {similarity_slots[0, 0, 1]:.4f}")
    print(f"     Audio slot 1 vs Image slot 0: {similarity_slots[0, 1, 0]:.4f}")
    print(f"     Audio slot 1 vs Image slot 1: {similarity_slots[0, 1, 1]:.4f}")
    print(f"   Batch 1:")
    print(f"     Audio slot 0 vs Image slot 0: {similarity_slots[1, 0, 0]:.4f}")
    print(f"     Audio slot 0 vs Image slot 1: {similarity_slots[1, 0, 1]:.4f}")
    print(f"     Audio slot 1 vs Image slot 0: {similarity_slots[1, 1, 0]:.4f}")
    print(f"     Audio slot 1 vs Image slot 1: {similarity_slots[1, 1, 1]:.4f}")
    
    B, N, C = aud_slots.shape
    assert N == 2, "STE sort implemented for num_slots=2"
    
    # Audio STE permutation (choose which audio slot is best)
    a0_best = similarity_slots[:, 0, :].max(dim=1).values
    a1_best = similarity_slots[:, 1, :].max(dim=1).values
    logits_a = (a1_best - a0_best) / 0.3
    p_a = torch.sigmoid(logits_a)
    
    print("\n3. Permutación de Audio:")
    print(f"   a0_best (max sim de slot 0): {a0_best}")
    print(f"   a1_best (max sim de slot 1): {a1_best}")
    print(f"   logits_a (diferencia escalada): {logits_a}")
    print(f"   p_a (probabilidad de que slot 1 sea mejor): {p_a}")
    
    P_a_soft = torch.stack([
        torch.stack([1 - p_a, p_a], dim=1),
        torch.stack([p_a, 1 - p_a], dim=1)
    ], dim=1)
    
    a_choose1 = (p_a > 0.5).long()
    P_a_hard = torch.zeros_like(P_a_soft)
    P_a_hard[torch.arange(B, device=P_a_soft.device), 0, a_choose1] = 1
    P_a_hard[torch.arange(B, device=P_a_soft.device), 1, 1 - a_choose1] = 1
    P_a = P_a_hard.detach() - P_a_soft.detach() + P_a_soft
    
    print(f"   a_choose1 (decisión hard): {a_choose1}")
    print(f"   Batch 0 P_a (matriz de permutación):\n{P_a[0]}")
    print(f"   Batch 1 P_a (matriz de permutación):\n{P_a[1]}")
    
    # Image STE permutation (choose which image slot is best)
    i0_best = similarity_slots[:, :, 0].max(dim=1).values
    i1_best = similarity_slots[:, :, 1].max(dim=1).values
    logits_i = (i1_best - i0_best) / 0.3
    p_i = torch.sigmoid(logits_i)
    
    print("\n4. Permutación de Imagen:")
    print(f"   i0_best (max sim de slot 0): {i0_best}")
    print(f"   i1_best (max sim de slot 1): {i1_best}")
    print(f"   logits_i (diferencia escalada): {logits_i}")
    print(f"   p_i (probabilidad de que slot 1 sea mejor): {p_i}")
    
    P_i_soft = torch.stack([
        torch.stack([1 - p_i, p_i], dim=1),
        torch.stack([p_i, 1 - p_i], dim=1)
    ], dim=1)
    
    i_choose1 = (p_i > 0.5).long()
    P_i_hard = torch.zeros_like(P_i_soft)
    P_i_hard[torch.arange(B, device=P_i_soft.device), 0, i_choose1] = 1
    P_i_hard[torch.arange(B, device=P_i_soft.device), 1, 1 - i_choose1] = 1
    P_i = P_i_hard.detach() - P_i_soft.detach() + P_i_soft
    
    print(f"   i_choose1 (decisión hard): {i_choose1}")
    print(f"   Batch 0 P_i (matriz de permutación):\n{P_i[0]}")
    print(f"   Batch 1 P_i (matriz de permutación):\n{P_i[1]}")
    
    # Reorder without mixing in forward
    aud_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['slots'])
    img_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['slots'])
    
    aud_slot_out['q_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['q'])
    img_slot_out['q_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['q'])
    
    aud_slot_out['attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['intra_attn'])
    img_slot_out['attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['intra_attn'])
    # === FIN DEL CÓDIGO REAL ===
    
    print("\n5. Verificación:")
    print("   Comparando reordenamiento usando P vs reordenamiento ideal...")
    
    # Calcular el reordenamiento "ideal" (basado en similitud máxima)
    # Para audio: el slot con mayor similitud máxima va primero
    a0_max_sim = similarity_slots[:, 0, :].max(dim=1).values  # Max sim de slot 0
    a1_max_sim = similarity_slots[:, 1, :].max(dim=1).values  # Max sim de slot 1
    
    # Determinar qué slot debe ir primero (el de mayor similitud)
    ideal_audio_perm = (a1_max_sim > a0_max_sim).long()  # 0 si slot 0 es mejor, 1 si slot 1 es mejor
    
    # Reordenamiento ideal para audio
    aud_slots_ideal = aud_slot_out['slots'].clone()
    for b in range(B):
        if ideal_audio_perm[b] == 1:
            # Intercambiar slots: slot 1 va primero
            aud_slots_ideal[b, 0], aud_slots_ideal[b, 1] = aud_slots_ideal[b, 1].clone(), aud_slots_ideal[b, 0].clone()
    
    # Para imagen: el slot con mayor similitud máxima va primero
    i0_max_sim = similarity_slots[:, :, 0].max(dim=1).values  # Max sim de slot 0
    i1_max_sim = similarity_slots[:, :, 1].max(dim=1).values  # Max sim de slot 1
    
    ideal_image_perm = (i1_max_sim > i0_max_sim).long()  # 0 si slot 0 es mejor, 1 si slot 1 es mejor
    
    # Reordenamiento ideal para imagen
    img_slots_ideal = img_slot_out['slots'].clone()
    for b in range(B):
        if ideal_image_perm[b] == 1:
            # Intercambiar slots: slot 1 va primero
            img_slots_ideal[b, 0], img_slots_ideal[b, 1] = img_slots_ideal[b, 1].clone(), img_slots_ideal[b, 0].clone()
    
    print(f"\n   Decisión ideal vs decisión P:")
    print(f"     Audio - Ideal: {ideal_audio_perm}, P (a_choose1): {a_choose1}")
    print(f"     Imagen - Ideal: {ideal_image_perm}, P (i_choose1): {i_choose1}")
    
    # Comparar resultados
    print(f"\n   Comparación de slots reordenados:")
    aud_diff_0 = torch.norm(aud_slot_out['slots_sorted'][0] - aud_slots_ideal[0])
    aud_diff_1 = torch.norm(aud_slot_out['slots_sorted'][1] - aud_slots_ideal[1])
    img_diff_0 = torch.norm(img_slot_out['slots_sorted'][0] - img_slots_ideal[0])
    img_diff_1 = torch.norm(img_slot_out['slots_sorted'][1] - img_slots_ideal[1])
    
    print(f"     Audio Batch 0 diferencia: {aud_diff_0:.6f}")
    print(f"     Audio Batch 1 diferencia: {aud_diff_1:.6f}")
    print(f"     Imagen Batch 0 diferencia: {img_diff_0:.6f}")
    print(f"     Imagen Batch 1 diferencia: {img_diff_1:.6f}")
    
    if aud_diff_0 < 1e-5 and aud_diff_1 < 1e-5 and img_diff_0 < 1e-5 and img_diff_1 < 1e-5:
        print("\n   ✓ ÉXITO: El reordenamiento usando P coincide con el reordenamiento ideal!")
    else:
        print("\n   ✗ ADVERTENCIA: Hay diferencias entre el reordenamiento usando P y el ideal.")
        print("     Esto puede ser normal si P usa soft permutation (STE).")
        
        # Mostrar diferencias más detalladas
        print(f"\n   Diferencias detalladas:")
        print(f"     Audio Batch 0 - slot 0:")
        print(f"       P: {aud_slot_out['slots_sorted'][0, 0][:5]}")
        print(f"       Ideal: {aud_slots_ideal[0, 0][:5]}")
    
    # Validar que las matrices de permutación son válidas
    print("\n6. Validación de matrices de permutación:")
    
    # Verificar que cada fila suma a 1
    P_a_row_sums = P_a.sum(dim=-1)
    P_i_row_sums = P_i.sum(dim=-1)
    print(f"   Sumas de filas P_a (deben ser ~1):")
    print(f"     Batch 0: fila 0={P_a_row_sums[0, 0]:.4f}, fila 1={P_a_row_sums[0, 1]:.4f}")
    print(f"     Batch 1: fila 0={P_a_row_sums[1, 0]:.4f}, fila 1={P_a_row_sums[1, 1]:.4f}")
    print(f"   Sumas de filas P_i (deben ser ~1):")
    print(f"     Batch 0: fila 0={P_i_row_sums[0, 0]:.4f}, fila 1={P_i_row_sums[0, 1]:.4f}")
    print(f"     Batch 1: fila 0={P_i_row_sums[1, 0]:.4f}, fila 1={P_i_row_sums[1, 1]:.4f}")
    
    # Verificar que cada columna suma a 1
    P_a_col_sums = P_a.sum(dim=1)
    P_i_col_sums = P_i.sum(dim=1)
    print(f"   Sumas de columnas P_a (deben ser ~1):")
    print(f"     Batch 0: col 0={P_a_col_sums[0, 0]:.4f}, col 1={P_a_col_sums[0, 1]:.4f}")
    print(f"     Batch 1: col 0={P_a_col_sums[1, 0]:.4f}, col 1={P_a_col_sums[1, 1]:.4f}")
    print(f"   Sumas de columnas P_i (deben ser ~1):")
    print(f"     Batch 0: col 0={P_i_col_sums[0, 0]:.4f}, col 1={P_i_col_sums[0, 1]:.4f}")
    print(f"     Batch 1: col 0={P_i_col_sums[1, 0]:.4f}, col 1={P_i_col_sums[1, 1]:.4f}")
    
    # Verificar que P_hard coincide con la decisión ideal
    print("\n7. Verificación de decisión hard vs ideal:")
    print(f"   Audio: ideal_perm={ideal_audio_perm}, a_choose1={a_choose1}")
    if (ideal_audio_perm == a_choose1).all():
        print("   ✓ Audio: La decisión hard coincide con la ideal")
    else:
        print("   ✗ Audio: La decisión hard NO coincide con la ideal")
        print(f"     Diferencia: {torch.abs(ideal_audio_perm.float() - a_choose1.float())}")
    
    print(f"   Imagen: ideal_perm={ideal_image_perm}, i_choose1={i_choose1}")
    if (ideal_image_perm == i_choose1).all():
        print("   ✓ Imagen: La decisión hard coincide con la ideal")
    else:
        print("   ✗ Imagen: La decisión hard NO coincide con la ideal")
        print(f"     Diferencia: {torch.abs(ideal_image_perm.float() - i_choose1.float())}")
    
    print("\n" + "=" * 70)
    print("Prueba completada!")
    print("=" * 70)


if __name__ == "__main__":
    test_slot_permutation_logic()

