import torch
import torch.nn.functional as F

def compute_matching_loss(intra_modal_attention_i, cross_modal_attention_ai, intra_modal_attention_a, cross_modal_attention_ia):
    # Cross-modal attention 
    # cross_modal_attention_ai # (B, 14, 14)
    # cross_modal_attention_ia # (B, 42)
    
   # Intra-modal attention 
    # intra_modal_attention_i # (B, 14, 14)
    # intra_modal_attention_a # (B, 42)
    
    # Cross-modal attention matching loss
    # L_match = ||ca^a,v - sg(ia^v,v)||_2^2 + ||ca^v,a - sg(ia^a,a)||_2^2
    
    # Stop-gradient operation   
    intra_modal_attention_i_sg = intra_modal_attention_i.detach()  # Stop gradient
    intra_modal_attention_a_sg = intra_modal_attention_a.detach()  # Stop gradient
    
    # Compute matching loss
    loss_match_aud_to_img = F.mse_loss(cross_modal_attention_ai, intra_modal_attention_i_sg, reduction='mean')
    loss_match_img_to_aud = F.mse_loss(cross_modal_attention_ia, intra_modal_attention_a_sg, reduction='mean')
    
    loss_match = loss_match_aud_to_img + loss_match_img_to_aud
    
    return loss_match

def compute_info_nce_loss(pv, pa, temperature=0.03, exclude_mask=None):
    """
    Contrastive loss with target slots (Eq. 7, page 4 of the paper).

    Args:
        pv: Tensor [B, D] target slots from the visual branch.
        pa: Tensor [B, D] target slots from the audio branch.
        temperature: float, temperature (default 0.03 as in the paper).
        exclude_mask: Optional boolean Tensor [B, B]. True = exclude this pair as a negative
                      (e.g., predicted false negatives). The diagonal is never excluded.

    Returns:
        loss: scalar tensor
    """
    assert pv.dim() == 2 and pa.dim() == 2 and pv.size(0) == pa.size(0), "Expected shape [B, D]."
    B = pv.size(0)

    # 1) Normalize (cosine similarity = dot product of L2 normalized vectors)
    v = F.normalize(pv, dim=-1)  # [B, D]
    a = F.normalize(pa, dim=-1)  # [B, D]

    # 2) Logits = cosine similarity / temperature
    logits_v2a = (v @ a.t()) / temperature          # [B, B]
    # logits_a2v = logits_v2a.t().contiguous()  # [B, B]
    logits_a2v = (a @ v.t()) / temperature  # [B, B]

    # 3) Optionally exclude false negatives (never exclude diagonal positives)
    if exclude_mask is not None:
        assert exclude_mask.shape == logits_v2a.shape and exclude_mask.dtype == torch.bool
        eye = torch.eye(B, dtype=torch.bool, device=exclude_mask.device)
        mask_no_pos = exclude_mask & (~eye)
        neg_inf = torch.finfo(logits_v2a.dtype).min
        logits_v2a = logits_v2a.masked_fill(mask_no_pos, neg_inf)
        logits_a2v = logits_a2v.masked_fill(mask_no_pos.t(), neg_inf)

    # 4) Labels: the correct positive is always the diagonal index
    labels = torch.arange(B, device=pv.device)

    # 5) Symmetric InfoNCE loss and average
    loss_v2a = F.cross_entropy(logits_v2a, labels)
    loss_a2v = F.cross_entropy(logits_a2v, labels)
    return 0.5 * (loss_v2a + loss_a2v)
    # return loss_v2a

def compute_info_nce_loss2(pv, pa, temperature=0.03, exclude_mask=None):
    """
    Contrastive loss with target slots following the exact paper formula (Eq. 7).
    
    This implements the formula:
    L_cotr = -(1/B) * Σ_{i=1}^B [log(s(p_i^v, p_i^a) / Σ_{j=1}^B s(p_i^v, p_j^a)) + 
                                  log(s(p_i^a, p_i^v) / Σ_{j=1}^B s(p_j^v, p_i^a))]
    
    where s(x, y) = exp(cos(x, y) / τ)

    Args:
        pv: Tensor [B, D] target slots from the visual branch.
        pa: Tensor [B, D] target slots from the audio branch.
        temperature: float, temperature τ (default 0.03 as in the paper).
        exclude_mask: Optional boolean Tensor [B, B]. True = exclude this pair as a negative
                      (e.g., predicted false negatives). The diagonal is never excluded.

    Returns:
        loss: scalar tensor
    """
    assert pv.dim() == 2 and pa.dim() == 2 and pv.size(0) == pa.size(0), "Expected shape [B, D]."
    B = pv.size(0)
    device = pv.device

    # 1) Normalize (cosine similarity = dot product of L2 normalized vectors)
    v = F.normalize(pv, dim=-1)  # [B, D]
    a = F.normalize(pa, dim=-1)  # [B, D]

    # 2) Compute cosine similarities
    cos_v2a = v @ a.t()  # [B, B] - cosine similarity between v_i and a_j
    cos_a2v = a @ v.t()  # [B, B] - cosine similarity between a_i and v_j

    # 3) Apply temperature and exp to get similarity function s(x, y) = exp(cos(x, y) / τ)
    s_v2a = torch.exp(cos_v2a / temperature)  # [B, B]
    s_a2v = torch.exp(cos_a2v / temperature)  # [B, B]

    # 4) Optionally exclude false negatives (never exclude diagonal positives)
    if exclude_mask is not None:
        assert exclude_mask.shape == s_v2a.shape and exclude_mask.dtype == torch.bool
        eye = torch.eye(B, dtype=torch.bool, device=device)
        mask_no_pos = exclude_mask & (~eye)
        s_v2a = s_v2a.masked_fill(mask_no_pos, 0.0)  # Set to 0 instead of -inf for exp
        s_a2v = s_a2v.masked_fill(mask_no_pos.t(), 0.0)

    # 5) Compute the two log terms for each sample i
    # Term 1: log(s(p_i^v, p_i^a) / Σ_{j=1}^B s(p_i^v, p_j^a))
    # Term 2: log(s(p_i^a, p_i^v) / Σ_{j=1}^B s(p_j^v, p_i^a))
    
    # Sum over j for each i (denominators)
    sum_v2a = s_v2a.sum(dim=1, keepdim=True)  # [B, 1] - sum over all a_j for each v_i
    sum_a2v = s_a2v.sum(dim=0, keepdim=True)  # [1, B] - sum over all v_j for each a_i
    
    # Extract diagonal elements (positive pairs)
    pos_v2a = torch.diag(s_v2a)  # [B] - s(p_i^v, p_i^a)
    pos_a2v = torch.diag(s_a2v)  # [B] - s(p_i^a, p_i^v)
    
    # Compute log terms with numerical stability
    eps = 1e-8
    term1 = torch.log(pos_v2a / (sum_v2a.squeeze() + eps))  # [B]
    term2 = torch.log(pos_a2v / (sum_a2v.squeeze() + eps))  # [B]
    
    # 6) Sum over all samples and average (multiply by -1/B)
    loss = -(term1 + term2).mean()  # [1] - scalar
    
    return loss

def compute_info_nce_loss3(pv: torch.Tensor,
                                 pa: torch.Tensor,
                                 temperature: float = 0.03) -> torch.Tensor:
    """
    InfoNCE simétrico (v->a y a->v) entre los target slots de imagen (pv) y audio (pa).

    Args:
        pv: Tensor [B, D] con los target slots de imagen.
        pa: Tensor [B, D] con los target slots de audio.
        tau: Temperatura (τ) para escalar similitudes.
        reduction: 'mean' | 'sum' | 'none'.

    Returns:
        loss: Escalar si reduction != 'none', o tensor [B] con la media de ambas direcciones por muestra.
    """
    assert pv.dim() == 2 and pa.dim() == 2, "pv y pa deben ser [B, D]"
    assert pv.shape == pa.shape, "pv y pa deben tener misma forma [B, D]"
    B = pv.size(0)

    # Cosine similarity: normalizamos a norma-2
    pv_n = F.normalize(pv, p=2, dim=1)
    pa_n = F.normalize(pa, p=2, dim=1)

    # Logits [B, B] = cos(pv_i, pa_j) / temperature
    logits = (pv_n @ pa_n.t()) / temperature

    # Pérdida v->a: fila i contra todas las columnas, target es el índice i (diagonal)
    loss_v2a = -torch.diag(F.log_softmax(logits, dim=1))  # [B]

    # Pérdida a->v: columna i contra todas las filas (equivale a softmax por dim=0)
    loss_a2v = -torch.diag(F.log_softmax(logits, dim=0))  # [B]

    # Promedio de las dos direcciones por muestra
    loss_per_sample = 0.5 * (loss_v2a + loss_a2v)  # [B]

    return loss_per_sample.mean()

def compute_divergence_loss(img_target_slot, img_offtarget_slot, aud_target_slot, aud_offtarget_slot):
    # Normalize slots for cosine similarity
    img_target_norm = F.normalize(img_target_slot, dim=-1)  # (B, 512)
    img_offtarget_norm = F.normalize(img_offtarget_slot, dim=-1)  # (B, 512)
    aud_target_norm = F.normalize(aud_target_slot, dim=-1)  # (B, 512)
    aud_offtarget_norm = F.normalize(aud_offtarget_slot, dim=-1)  # (B, 512)
    
    # Compute cosine similarity between target and off-target slots
    # Using einsum for batch-wise dot product
    img_cos_sim = torch.einsum('bd,bd->b', img_target_norm, img_offtarget_norm)  # (B,)
    aud_cos_sim = torch.einsum('bd,bd->b', aud_target_norm, aud_offtarget_norm)  # (B,)
    
    # Take max between 0 and cosine similarity
    div_loss_img = F.relu(img_cos_sim).mean() 
    div_loss_aud = F.relu(aud_cos_sim).mean()
    
    return div_loss_img + div_loss_aud

def compute_reconstruction_loss(img_orig, aud_orig, img_recon, aud_recon):
    """
    Compute slot reconstruction loss as per the paper:
    Lrecon = ||v - gv({pv; rv})||₂² + ||a - ga({pa; ra})||₂²
    
    Args:
        img_orig: Original image embedding (v in the paper)
        aud_orig: Original audio embedding (a in the paper) 
        img_recon: Reconstructed image features from decoder gv({pv; rv})
        aud_recon: Reconstructed audio features from decoder ga({pa; ra})
    
    Returns:
        reconstruction_loss: L2 reconstruction loss
    """
    # L2 norm squared for image reconstruction
    img_recon_loss = F.mse_loss(img_orig, img_recon, reduction='mean')
    
    # L2 norm squared for audio reconstruction  
    aud_recon_loss = F.mse_loss(aud_orig, aud_recon, reduction='mean')
    
    # Total reconstruction loss
    reconstruction_loss = img_recon_loss + aud_recon_loss
    
    return reconstruction_loss

def compute_loss(img_slot_out, aud_slot_out, args, mode='train'):

    if mode == 'train':
        loss_recon = args.lambda_recon * compute_reconstruction_loss(
            img_slot_out['emb'], # Original image embedding
            aud_slot_out['emb'], # Original audio embedding
            img_slot_out['emb_rec'],  # Reconstructed image embedding
            aud_slot_out['emb_rec']   # Reconstructed audio embedding
        )
        
        loss_match = args.lambda_match * compute_matching_loss(
            img_slot_out['intra_attn'][:, 0, :], # Target image intra-modal attention
            img_slot_out['cross_attn'][:, 0, :], # Target audio-image cross-modal attention
            aud_slot_out['intra_attn'][:, 0, :], # Target audio intra-modal attention
            aud_slot_out['cross_attn'][:, 0, :], # Target image-audio cross-modal attention
            )
        
        loss_div = args.lambda_div * compute_divergence_loss(
            img_slot_out['slots'][:, 0, :], # Target image slot
            img_slot_out['slots'][:, 1, :], # Off-target image slot
            aud_slot_out['slots'][:, 0, :], # Target audio slot   
            aud_slot_out['slots'][:, 1, :]  # Off-target audio slot
        )
        
        loss_info_nce = args.lambda_info_nce * compute_info_nce_loss3(
            img_slot_out['slots'][:, 0, :], # Target image slot
            aud_slot_out['slots'][:, 0, :], # Target audio slot
            temperature=args.tau # Temperature for InfoNCE loss
        )

        return loss_info_nce, loss_match, loss_div, loss_recon

    else:
        
        loss_info_nce = args.lambda_info_nce * compute_info_nce_loss3(
            img_slot_out['slots'][:, 0, :], # Target image slot
            aud_slot_out['slots'][:, 0, :], # Target audio slot
            temperature=args.tau # Temperature for InfoNCE loss
        )

        return loss_info_nce