import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import copy
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-like attention."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class EZVSL(nn.Module):
    def __init__(self, tau, dim, args):
        super(EZVSL, self).__init__()
        self.tau = tau
        self.args = args

        if self.args.imagenet_pretrain == 'True':
            pretrained = True
        else:
            pretrained = False
        # Vision model
        self.imgnet = resnet18(pretrained=pretrained)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_conv1d = nn.Conv1d(512, dim, kernel_size=1)
        
        if self.args.visual_dropout == 'True':
            # Heavy visual dropout (SLAVC-style)
            self.visual_dropout = nn.Dropout(p=self.args.visual_dropout_ratio)

        # Audio model
        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.avgpool = nn.Identity()
        self.audnet.fc = nn.Identity()
        self.aud_conv1d = nn.Conv1d(512, dim, kernel_size=1)
        # self.aud_proj = nn.Linear(512, dim)

        # Slot Attention
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim)) 
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim)) 
        
        if self.args.n_attention_modules == 1:
            self.slot_attention = SlotAttention(
                num_slots=args.num_slots,
                args=args,
                dim=dim,
                iters = args.iters,
                eps = args.eps, 
                hidden_dim = args.hidden_dim
            )
        else:
            if self.args.slot_clone == 'True':
                self.image_slot_attention = SlotAttention(
                    num_slots=args.num_slots,
                    args=args,
                    dim_in=512,
                    dim_out=dim,
                    iters = args.iters,
                    eps = args.eps, 
                    hidden_dim = args.hidden_dim
                )
                self.audio_slot_attention = copy.deepcopy(self.image_slot_attention)
            else:
                self.image_slot_attention = SlotAttention(
                    num_slots=args.num_slots,
                    args=args,
                    dim_in=512,
                    dim_out=dim,
                    iters = args.iters,
                    eps = args.eps, 
                    hidden_dim = args.hidden_dim
                )
                self.audio_slot_attention = SlotAttention(
                    num_slots=args.num_slots,
                    args=args,
                    dim_in=512,
                    dim_out=dim,
                    iters = args.iters,
                    eps = args.eps, 
                    hidden_dim = args.hidden_dim
                )

        self.img_slot_decoder = SlotDecoder(slot_dim=args.num_slots * dim, hidden_dim=dim)
        self.aud_slot_decoder = SlotDecoder(slot_dim=args.num_slots * dim, hidden_dim=dim)
        
        # Positional encoding for image (spatial dimension) and audio (temporal dimension)
        # Both use 512-dim embeddings from resnet
        if args.use_pos_encoding_img == 'True':
            self.img_pos_encoding = PositionalEncoding(d_model=512, max_len=100)
        else:
            self.img_pos_encoding = None
            
        if args.use_pos_encoding_aud == 'True':
            self.aud_pos_encoding = PositionalEncoding(d_model=512, max_len=100)
        else:
            self.aud_pos_encoding = None
                
        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_conv1d, self.aud_conv1d]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def max_xmil_loss(self, img, aud):
        B = img.shape[0]
        Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        labels = torch.arange(B).long().to(img.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)
        return loss, Slogits

    def forward(self, image, audio):
        # Image
        img = self.imgnet(image).unflatten(1, (512, 49))
        
        # Apply positional encoding to image embeddings (spatial dimension) right after resnet
        if self.img_pos_encoding is not None:
            # Permute to (B, seq_len, dim) format for positional encoding: (B, 512, 49) -> (B, 49, 512)
            img_seq = img.contiguous().permute(0, 2, 1)  # (B, 49, 512) - spatial dimension
            img_seq = self.img_pos_encoding(img_seq)
            # Permute back to original format: (B, 49, 512) -> (B, 512, 49)
            img = img_seq.permute(0, 2, 1)  # (B, 512, 49)
        
        if self.args.visual_dropout == 'True':
            # Apply dropout to the image features
            img = self.visual_dropout(img)
        
        img_maxpool = img.max(dim=-1).values
        img_maxpool = self.img_conv1d(img_maxpool.unsqueeze(-1)).squeeze(-1)
        img_maxpool = nn.functional.normalize(img_maxpool, dim=1)

        # Audio
        aud = self.audnet(audio).unflatten(1, (512, 9, 7))
        aud_temp = aud.clone().max(dim=2).values # Max-Pooling over frequency dimension
        
        # Apply positional encoding to audio embeddings (temporal dimension) right after frequency pooling
        if self.aud_pos_encoding is not None:
            # Permute to (B, seq_len, dim) format for positional encoding: (B, 512, 7) -> (B, 7, 512)
            aud_seq = aud_temp.contiguous().permute(0, 2, 1)  # (B, 7, 512) - temporal dimension
            aud_seq = self.aud_pos_encoding(aud_seq)
            # Permute back to original format: (B, 7, 512) -> (B, 512, 7)
            aud_temp = aud_seq.permute(0, 2, 1)  # (B, 512, 7)
        
        aud_pool = aud_temp.clone().max(dim=-1).values # Max-Pooling over temporal dimension
        aud_pool_proj = self.aud_conv1d(aud_pool.unsqueeze(-1)).squeeze(-1)
        aud_pool_proj = nn.functional.normalize(aud_pool_proj, dim=1)
        
        b, n, d, device, dtype = *aud_temp.shape, aud_temp.device, aud_temp.dtype 
        
        mu = self.slots_mu.expand(b, self.args.num_slots, -1) 
        sigma = self.slots_logsigma.exp().expand(b, self.args.num_slots, -1) 
        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype) 
        slots_image = slots.clone()
        slots_audio = slots.clone()

        # Slot Attention
        # Permute to (B, seq_len, dim) format for slot attention
        img_seq = img.contiguous().permute(0, 2, 1)  # (B, 49, 512) - spatial dimension
        aud_seq = aud_temp.contiguous().permute(0, 2, 1)  # (B, 9, 512) - temporal dimension
        
        if self.args.n_attention_modules == 1:
            img_slot_out = self.slot_attention(img_seq, shared_init_slots=slots_image)
            aud_slot_out = self.slot_attention(aud_seq, shared_init_slots=slots_audio)
        else:
            img_slot_out = self.image_slot_attention(img_seq, shared_init_slots=slots_image)
            aud_slot_out = self.audio_slot_attention(aud_seq, shared_init_slots=slots_audio)

        img_recon = self.img_slot_decoder(img_slot_out['slots'].contiguous().view((img_slot_out['slots'].shape[0], -1)))
        aud_recon = self.aud_slot_decoder(aud_slot_out['slots'].contiguous().view((aud_slot_out['slots'].shape[0], -1)))
        
        aud_slot_out['embedding_original'] = aud_pool
        img_slot_out['embedding_original'] = img
        
        aud_slot_out['emb'] = aud_pool_proj
        img_slot_out['emb'] = img_maxpool

        aud_slot_out['emb_rec'] = aud_recon
        img_slot_out['emb_rec'] = img_recon

        return aud_slot_out, img_slot_out

class SlotAttention(nn.Module): 
    def __init__(self, num_slots, dim_in, dim_out, iters = 3, eps = 1e-8, hidden_dim = 128, args = None): 
        super().__init__()
        self.args = args
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_slots = num_slots 
        self.iters = iters 
        self.eps = eps 
        self.scale = dim_out ** -0.5 

        self.w_q = nn.Linear(dim_out, dim_out, bias=(self.args.w_bias == 'True')) 
        self.w_k = nn.Linear(dim_in, dim_out, bias=(self.args.w_bias == 'True')) 
        self.w_v = nn.Linear(dim_in, dim_out, bias=(self.args.w_bias == 'True')) 

        if self.args.add_gru == 'True':
            self.gru = nn.GRUCell(dim_out, dim_out) 
            
        if self.args.add_mlp == 'True':
            hidden_dim = max(dim_out, hidden_dim) 
            self.mlp = nn.Sequential( nn.Linear(dim_out, hidden_dim), nn.ReLU(inplace = True), nn.Linear(hidden_dim, dim_out) ) 
            
        self.LayerNorm_input = nn.LayerNorm(dim_in) 
        self.LayerNorm_slots = nn.LayerNorm(dim_out) 
        self.LayerNorm_pre_ff = nn.LayerNorm(dim_out) 
        
    def forward(self, inputs, shared_init_slots = None): 
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype 
        
        inputs = self.LayerNorm_input(inputs) 
        k, v = self.w_k(inputs), self.w_v(inputs) 
        slots = shared_init_slots.clone()
        
        # Store debug values for each iteration
        debug_dots = []
        debug_attn_pre_norm = []
        debug_attn = []
        
        for i in range(self.iters): 
            slots_prev = slots.clone()
            slots = self.LayerNorm_slots(slots)  
            q = self.w_q(slots) 
            
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale 
            debug_dots.append(dots.detach().clone())
            
            attn_pre_norm = dots.softmax(dim=1) 
            debug_attn_pre_norm.append(attn_pre_norm.detach().clone())
            
            attn = attn_pre_norm / (attn_pre_norm.sum(dim=-1, keepdim=True) + self.eps )
            debug_attn.append(attn.detach().clone())
            
            updates = torch.einsum('bjd,bij->bid', v, attn) 
            
            # slots = self.gru( updates.reshape(-1, d), slots_prev.reshape(-1, d) ) 
            if self.args.add_gru == 'True':
                slots = self.gru( updates.flatten(0, 1), slots_prev.flatten(0, 1) ) 
            else:
                slots = updates.flatten(0, 1)
            
            if self.args.add_mlp == 'True':
                slots = slots + self.mlp(self.LayerNorm_pre_ff(slots)) 
            
            slots = slots.unflatten(0, (b, 2))
            # slots = slots.reshape(b, -1, d) 

        slots_data = { 
            'slots': slots, 
            'q': q, 
            'k': k, 
            'intra_attn': attn,
            'debug_dots': debug_dots,
            'debug_attn_pre_norm': debug_attn_pre_norm,
            'debug_attn': debug_attn
        } 
        
        return slots_data

class SlotAttention2(nn.Module): 
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, args = None): 
        super().__init__()        
        self.args = args
        self.dim = dim 
        self.num_slots = num_slots 
        self.iters = iters 
        self.eps = eps 
        self.scale = dim ** -0.5 
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, 512)) 
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, 512)) 

        # if self.args.slots_no_W == 'True':
        #     self.w_q = nn.Identity() 
        #     self.w_k = nn.Identity() 
        #     self.w_v = nn.Identity() 
        # else:
        self.w_q = nn.Linear(dim, dim) 
        self.w_k = nn.Linear(dim, dim) 
        self.w_v = nn.Linear(dim, dim) 

        self.gru = nn.GRUCell(dim, dim) 
        hidden_dim = max(dim, hidden_dim) 
        
        self.mlp = nn.Sequential( nn.Linear(dim, hidden_dim), nn.ReLU(inplace = True), nn.Linear(hidden_dim, dim) ) 
        self.LayerNorm_input = nn.LayerNorm(dim) 
        self.LayerNorm_slots = nn.LayerNorm(dim) 
        self.LayerNorm_pre_ff = nn.LayerNorm(dim) 
        
    def forward(self, inputs_audio, inputs_image, num_slots = None, shared_init_slots = None): 
        b, n_audio, d_audio, device, dtype = *inputs_audio.shape, inputs_audio.device, inputs_audio.dtype 
        b, n_image, d_image, device, dtype = *inputs_image.shape, inputs_image.device, inputs_image.dtype 

        mu = self.slots_mu.expand(b, self.args.num_slots, -1) 
        sigma = self.slots_logsigma.exp().expand(b, self.args.num_slots, -1) 
        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype) 
        
        inputs_audio = self.LayerNorm_input(inputs_audio) 
        inputs_image = self.LayerNorm_input(inputs_image) 

        k_audio, v_audio = self.w_k(inputs_audio), self.w_v(inputs_audio) 
        k_image, v_image = self.w_k(inputs_image), self.w_v(inputs_image) 

        slots_audio = slots.clone()
        slots_image = slots.clone()
        
        for i in range(self.iters): 
            slots_prev_audio = slots_audio 
            slots_prev_image = slots_image 

            slots_audio = self.LayerNorm_slots(slots_audio) 
            slots_image = self.LayerNorm_slots(slots_image) 

            q_audio = self.w_q(slots_audio) 
            q_image = self.w_q(slots_image) 

            
            dots_audio = torch.einsum('bid,bjd->bij', q_audio, k_audio) * self.scale 
            dots_image = torch.einsum('bid,bjd->bij', q_image, k_image) * self.scale 

            attn_pre_norm_audio = dots_audio.softmax(dim=1) 
            attn_pre_norm_image = dots_image.softmax(dim=1) 

            attn_audio = attn_pre_norm_audio / (attn_pre_norm_audio.sum(dim=-1, keepdim=True) + self.eps )
            attn_image = attn_pre_norm_image / (attn_pre_norm_image.sum(dim=-1, keepdim=True) + self.eps )
               
            updates_audio = torch.einsum('bjd,bij->bid', v_audio, attn_audio) 
            updates_image = torch.einsum('bjd,bij->bid', v_image, attn_image) 
            
            slots_audio = self.gru( updates_audio.reshape(-1, d_audio), slots_prev_audio.reshape(-1, d_audio) ) 
            slots_audio = slots_audio.reshape(b, -1, d_audio) 
            slots_audio = slots_audio + self.mlp(self.LayerNorm_pre_ff(slots_audio)) 

            slots_image = self.gru( updates_image.reshape(-1, d_image), slots_prev_image.reshape(-1, d_image) ) 
            slots_image = slots_image.reshape(b, -1, d_image) 
            slots_image = slots_image + self.mlp(self.LayerNorm_pre_ff(slots_image)) 
        
        cross_attn_audio_image = torch.einsum('bid,bjd->bij', q_audio, k_image) * self.scale 
        cross_attn_audio_image = cross_attn_audio_image.softmax(dim=1)
        cross_attn_audio_image = cross_attn_audio_image / (cross_attn_audio_image.sum(dim=-1, keepdim=True) + self.eps )

        cross_attn_image_audio = torch.einsum('bid,bjd->bij', q_image, k_audio) * self.scale 
        cross_attn_image_audio = cross_attn_image_audio.softmax(dim=1)
        cross_attn_image_audio = cross_attn_image_audio / (cross_attn_image_audio.sum(dim=-1, keepdim=True) + self.eps )

        slots_data_audio = { 'slots': slots_audio, 'q': q_audio, 'k': k_audio, 'intra_attn': attn_audio, 'cross_attn': cross_attn_image_audio } 
        slots_data_image = { 'slots': slots_image, 'q': q_image, 'k': k_image, 'intra_attn': attn_image, 'cross_attn': cross_attn_audio_image } 
        
        return slots_data_audio, slots_data_image

class SlotDecoder(nn.Module):
    def __init__(self, slot_dim, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # MLP with a small number of layers and ReLU activations
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, slots):
        x = self.mlp(slots)
        return x