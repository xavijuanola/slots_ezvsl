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
        loss_recon = compute_reconstruction_loss(
            img_slot_out['emb'], # Original image embedding
            aud_slot_out['emb'], # Original audio embedding
            img_slot_out['emb_rec'],  # Reconstructed image embedding
            aud_slot_out['emb_rec']   # Reconstructed audio embedding
        )
        
        loss_match = compute_matching_loss(
            img_slot_out['intra_attn'][:, 0, :], # Target image intra-modal attention
            img_slot_out['cross_attn'][:, 0, :], # Target audio-image cross-modal attention
            aud_slot_out['intra_attn'][:, 0, :], # Target audio intra-modal attention
            aud_slot_out['cross_attn'][:, 0, :], # Target image-audio cross-modal attention
            )
        
        loss_div = compute_divergence_loss(
            img_slot_out['slots'][:, 0, :], # Target image slot
            img_slot_out['slots'][:, 1, :], # Off-target image slot
            aud_slot_out['slots'][:, 0, :], # Target audio slot   
            aud_slot_out['slots'][:, 1, :]  # Off-target audio slot
        )
        
        loss_info_nce = compute_info_nce_loss(
            img_slot_out['slots'][:, 0, :], # Target image slot
            aud_slot_out['slots'][:, 0, :], # Target audio slot
            temperature=args.tau # Temperature for InfoNCE loss
        )

        return loss_info_nce, loss_match, loss_div, loss_recon

    else:
        
        loss_info_nce = compute_info_nce_loss(
            img_slot_out['slots'][:, 0, :], # Target image slot
            aud_slot_out['slots'][:, 0, :], # Target audio slot
            temperature=args.tau # Temperature for InfoNCE loss
        )

        return loss_info_nce