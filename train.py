import os
import matplotlib.pyplot as plt
from matplotlib import patches
import tqdm
import argparse
import builtins
import time
import numpy as np
from torch.optim import *

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist
import wandb

import utils
from model import EZVSL
from losses import compute_loss, permutation_consistency_loss
from datasets import get_train_dataset, get_test_dataset, get_train_test_dataset
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR


def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Inverse normalization for ImageNet normalized tensors
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean values used in normalization
        std: Std values used in normalization
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ezvsl_vggss', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--trainset', default='vggss_144k', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss_144k', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='/media/y/datasets/vggsound/train', type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='/media/v/vggss', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='metadata/vggss_annotations.json', type=str)

    # ez-vsl hyper-params
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # slot attention hyper-params
    parser.add_argument('--add_gru', default='True', type=str, help='["False", "True"]')
    parser.add_argument('--add_mlp', default='True', type=str, help='["False", "True"]')
    parser.add_argument('--w_bias', default='True', type=str, help='whether to use bias in the attention weights')
    parser.add_argument('--n_attention_modules', default=2, type=int, help='number of attention modules')
    parser.add_argument('--slot_clone', default='True', type=str, help='whether to clone slot attention module (same W initialization)')
    parser.add_argument('--num_slots', default=2, type=int)
    parser.add_argument('--slots_maxsim', default='False', type=str, help='Wether to select target slot based on maximum similarity')
    parser.add_argument('--iters', default=5, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--slots_no_W', default='False', type=str)

    parser.add_argument('--lambda_info_nce', default=1.0, type=float)
    parser.add_argument('--lambda_match', default=100.0, type=float)
    parser.add_argument('--lambda_div', default=0.1, type=float)
    parser.add_argument('--lambda_recon', default=0.1, type=float)
    parser.add_argument('--use_perm_reg', default='False', type=str, help='Use permutation consistency regularizer')
    parser.add_argument('--lambda_perm', default=0.01, type=float, help='Weight for permutation consistency regularizer')

    # training/evaluation parameters
    parser.add_argument('--debug', type=str, default='True', help='debug mode')
    parser.add_argument('--log_debug_attentions', type=str, default='False', help='log debug attentions')
    parser.add_argument('--imagenet_pretrain', type=str, default='True', help='list of imagenet pretrain files')
    parser.add_argument('--visual_dropout', type=str, default='False', help='visual dropout')
    parser.add_argument('--visual_dropout_ratio', type=float, default=0.7, help='visual dropout ratio')
    parser.add_argument('--use_pos_encoding_img', type=str, default='False', help='use positional encoding for image embeddings (spatial dimension)')
    parser.add_argument('--use_pos_encoding_aud', type=str, default='False', help='use positional encoding for audio embeddings (temporal dimension)')
    parser.add_argument('--image_augmentations', type=str, default='ssltie', help='["No", "ssltie", "ezvsl"]')
    parser.add_argument('--audio_augmentations', type=str, default='ssltie', help='["No", "ssltie"]')
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument("--init_lr", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument('--wandb', type=str, default='True', help='use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='ezvsl-slots', help='wandb project name')

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


def main(args):
    mp.set_start_method('spawn')
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model dir
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    args.experiment_name = f'{args.experiment_name}_{timestamp}'
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # Initialize wandb (only on rank 0)
    if args.wandb == 'True':
        if args.debug == 'True':
            wandb.init(
                project=f'{args.wandb_project}_debug',
                name=f'{args.experiment_name}_debug',
                config=vars(args)
            )
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args)
            )

    # Create model
    model = EZVSL(args.tau, args.out_dim, args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    # print(model)

    # Optimizer
    # optimizer, _ = utils.build_optimizer_and_scheduler_adamW(model, args)
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.init_lr, betas=(0.9, 0.999)) # Better for weight decay

    warmup_epochs = 0
    total_epochs = args.epochs

    # Warmup scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

    # ReduceLROnPlateau scheduler (monitor validation loss)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,             # how much to reduce LR by
        patience=5,             # how many epochs to wait before reducing LR
        min_lr=1e-8,
        verbose=True
    )

    # Use warmup scheduler directly (no SequentialLR needed)
    scheduler = plateau_scheduler

    # Store plateau_scheduler separately for manual stepping after warmup
    # The main scheduler object is still warmup_scheduler, but store both for use in main training loop
    # scheduler = SequentialLR(optimizer, [warmup_scheduler], milestones=[warmup_epochs])

    # Resume if possible
    start_epoch, best_cIoU, best_Auc = 0, 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cpu')
        start_epoch, best_cIoU, best_Auc = ckp['epoch'], ckp['best_cIoU'], ckp['best_Auc']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        if 'scheduler' in ckp:  # Add this check
            scheduler.load_state_dict(ckp['scheduler'])
        if 'plateau_scheduler' in ckp:
            plateau_scheduler.load_state_dict(ckp['plateau_scheduler'])
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    # Dataloaders
    if args.testset == 'vggss_144k':
        traindataset, testdataset = get_train_test_dataset(args)
    else:
        traindataset = get_train_dataset(args)
        testdataset = get_test_dataset(args)

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, sampler=None, drop_last=False,
        persistent_workers=args.workers > 0)

    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        persistent_workers=args.workers > 0)
    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    if args.testset == 'vggss_144k':
        loss_info_nce = validate(test_loader, model, args, start_epoch)
        print(f'    ----Validation epoch {start_epoch}----')
        print(f'    Info NCE Loss (epoch {start_epoch}): {loss_info_nce:.4f}')
    else:
        cIoU, auc = validate_vggss(test_loader, model, args, start_epoch)
        print(f'    ----Validation epoch {start_epoch}----')
        print(f'    cIoU (epoch {start_epoch}): {cIoU:.4f}')
        print(f'    AUC (epoch {start_epoch}): {auc:.4f}')

    # Log visualizations for initial validation (only on rank 0)
    if args.wandb == 'True':
        if hasattr(args, 'train_log_files') and args.train_log_files:
            # print(f'    Logging initial train visualizations for epoch {start_epoch}...')
            log_file_visualizations(traindataset, model, args.train_log_files, 'train', start_epoch, args)
        if hasattr(args, 'val_log_files') and args.val_log_files:
            # print(f'    Logging initial validation visualizations for epoch {start_epoch}...')
            log_file_visualizations(testdataset, model, args.val_log_files, 'val', start_epoch, args)
        
    best_loss_info_nce = loss_info_nce
    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, scheduler, epoch, args)

        # Step the scheduler after each epoch
        if epoch < warmup_epochs:
            # During warmup, step the warmup scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Warmup epoch {epoch+1}/{warmup_epochs}, Current learning rate: {current_lr:.6f}")
        else:
            # After warmup, step the plateau scheduler with validation loss
            plateau_scheduler.step(loss_info_nce)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Plateau scheduler epoch {epoch+1}, Current learning rate: {current_lr:.6f}")

        # Evaluate
        print(f'    ----Validation epoch {epoch}----')
        if args.testset == 'vggss_144k':
            loss_info_nce = validate(test_loader, model, args, epoch)
            print(f'    Info NCE Loss (epoch {epoch}): {loss_info_nce:.4f}')
        else:
            cIoU, auc = validate_vggss(test_loader, model, args, epoch)
            print(f'    cIoU (epoch {epoch}): {cIoU:.4f}')
            print(f'    AUC (epoch {epoch}): {auc:.4f}')

        # Log visualizations for specified files after epoch completion (only on rank 0)
        if args.wandb == 'True':
            if hasattr(args, 'train_log_files') and args.train_log_files:
                print(f'    Logging train visualizations for epoch {epoch}...')
                log_file_visualizations(traindataset, model, args.train_log_files, 'train', epoch, args)
            if hasattr(args, 'val_log_files') and args.val_log_files:
                print(f'    Logging validation visualizations for epoch {epoch}...')
                log_file_visualizations(testdataset, model, args.val_log_files, 'val', epoch, args)

        # # Log validation metrics to wandb (only on rank 0)
        # if args.wandb a== 'True':
        #     if args.testset == 'vggss_144k':
        #         wandb.log({
        #             'val/Info NCE Loss': loss_info_nce.item(),
        #             'val/avg_loss': avg_loss.avg,
        #             'val/epoch': epoch
        #         })
        #     else:
        #         wandb.log({
        #             'val/cIoU': cIoU,
        #             'val/AUC': auc,
        #             'val/best_cIoU': best_cIoU,
        #             'val/best_AUC': best_Auc,
        #             'epoch': epoch + 1
        #         })
        
        # Checkpoint
        if args.debug != 'True':
            if args.rank == 0:
                ckp = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'plateau_scheduler': plateau_scheduler.state_dict(),  # Add plateau scheduler state
                    'epoch': epoch+1,
                    'best_cIoU': best_cIoU,
                    'best_Auc': best_Auc}
                
                # Save latest checkpoint
                torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
                print(f"Latest model saved to {model_dir}")
                
                # Save epoch-specific checkpoint
                epoch_ckp_path = os.path.join(model_dir, f'epoch_{epoch+1:03d}.pth')
                torch.save(ckp, epoch_ckp_path)
                print(f"Epoch {epoch+1} model saved to {epoch_ckp_path}")
            
            if args.testset == 'vggss_144k':
                if loss_info_nce >= best_loss_info_nce:
                    best_loss_info_nce = loss_info_nce
                    if args.rank == 0:
                        torch.save(ckp, os.path.join(model_dir, 'best.pth'))
                        print(f"Best model saved to {model_dir}")
            else:
                if cIoU >= best_cIoU:
                    best_cIoU, best_Auc = cIoU, auc
                    if args.rank == 0:
                        torch.save(ckp, os.path.join(model_dir, 'best.pth'))
                        print(f"Best model saved to {model_dir}")
                        epoch_ckp_path = os.path.join(model_dir, f'epoch_{epoch+1:03d}.pth')
                        torch.save(ckp, epoch_ckp_path)
                        print(f"Epoch {epoch+1} model saved to {epoch_ckp_path}")
    # Finish wandb run
    if args.wandb == 'True' and args.rank == 0:
        wandb.finish()

def train(train_loader, model, optimizer, scheduler, epoch, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    # Initialize loss averages for the epoch
    avg_total_loss = AverageMeter('Total Loss', ':.4f')
    avg_info_nce_loss = AverageMeter('Info NCE Loss', ':.4f')
    avg_match_loss = AverageMeter('Matching Loss', ':.4f') 
    avg_div_loss = AverageMeter('Divergence Loss', ':.4f')
    avg_recon_loss = AverageMeter('Reconstruction Loss', ':.4f')

    end = time.time()
    pbar = tqdm.tqdm(train_loader, desc=f'Training epoch {epoch+1}')
    
    # Add training step counter
    train_step = epoch * len(train_loader)  # Start from epoch beginning
    
    for i, (image, spec, bboxes, filename) in enumerate(pbar):
        B = image.shape[0]
        data_time.update(time.time() - end)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        aud_slot_out, img_slot_out = model(image.float(), spec.float())
        
        if args.slots_maxsim == 'True':
            # Normalize per-slot vectors on channel dim
            aud_slots = F.normalize(aud_slot_out['slots'], dim=2)
            img_slots = F.normalize(img_slot_out['slots'], dim=2)
            
            # Pairwise similarity S[b, n_a, n_i]
            similarity_slots = torch.einsum('bnc,bmc->bnm', aud_slots, img_slots)

            B, N, C = aud_slots.shape
            assert N == 2, "STE sort implemented for num_slots=2"

            # Audio STE permutation (choose which audio slot is best)
            a0_best = similarity_slots[:, 0, :].max(dim=1).values
            a1_best = similarity_slots[:, 1, :].max(dim=1).values
            logits_a = (a1_best - a0_best) / 0.3
            p_a = torch.sigmoid(logits_a)
            P_a_soft = torch.stack([
                torch.stack([1 - p_a, p_a], dim=1),
                torch.stack([p_a, 1 - p_a], dim=1)
            ], dim=1)
            a_choose1 = (p_a > 0.5).long()
            P_a_hard = torch.zeros_like(P_a_soft)
            P_a_hard[torch.arange(B, device=P_a_soft.device), 0, a_choose1] = 1
            P_a_hard[torch.arange(B, device=P_a_soft.device), 1, 1 - a_choose1] = 1
            P_a = P_a_hard.detach() - P_a_soft.detach() + P_a_soft

            # Image STE permutation (choose which image slot is best)
            i0_best = similarity_slots[:, :, 0].max(dim=1).values
            i1_best = similarity_slots[:, :, 1].max(dim=1).values
            logits_i = (i1_best - i0_best) / 0.3
            p_i = torch.sigmoid(logits_i)
            P_i_soft = torch.stack([
                torch.stack([1 - p_i, p_i], dim=1),
                torch.stack([p_i, 1 - p_i], dim=1)
            ], dim=1)
            i_choose1 = (p_i > 0.5).long()
            P_i_hard = torch.zeros_like(P_i_soft)
            P_i_hard[torch.arange(B, device=P_i_soft.device), 0, i_choose1] = 1
            P_i_hard[torch.arange(B, device=P_i_soft.device), 1, 1 - i_choose1] = 1
            P_i = P_i_hard.detach() - P_i_soft.detach() + P_i_soft

            # Reorder without mixing in forward
            aud_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['slots'])
            img_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['slots'])
            
            aud_slot_out['q_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['q'])
            img_slot_out['q_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['q'])
            
            aud_slot_out['attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['intra_attn'])
            img_slot_out['attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['intra_attn'])

            # Optional: permutation consistency regularizer (encourage P_soft ≈ P_hard)
            if args.use_perm_reg == 'True':
                y_a = a_choose1.float().detach()
                y_i = i_choose1.float().detach()
                loss_perm = permutation_consistency_loss(p_a, y_a, p_i, y_i)
            else:
                loss_perm = None

        else:
            aud_slot_out['slots_sorted'] = aud_slot_out['slots']
            img_slot_out['slots_sorted'] = img_slot_out['slots']

            aud_slot_out['q_sorted'] = aud_slot_out['q']
            img_slot_out['q_sorted'] = img_slot_out['q']
            
            aud_slot_out['attn_sorted'] = aud_slot_out['intra_attn']
            img_slot_out['attn_sorted'] = img_slot_out['intra_attn']
        
        cross_modal_attention_ai = torch.einsum('bid,bjd->bij', aud_slot_out['q_sorted'], img_slot_out['k']) * (512 ** -0.5)
        cross_modal_attention_ai = cross_modal_attention_ai.softmax(dim=1) + 1e-8
        h = w = int(cross_modal_attention_ai.shape[-1] ** 0.5)
        cross_modal_attention_ai = (cross_modal_attention_ai / cross_modal_attention_ai.sum(dim=-1, keepdim=True))

        cross_modal_attention_ia = torch.einsum('bid,bjd->bij', img_slot_out['q_sorted'], aud_slot_out['k']) * (512 ** -0.5)
        cross_modal_attention_ia = cross_modal_attention_ia.softmax(dim=1) + 1e-8
        cross_modal_attention_ia = (cross_modal_attention_ia / cross_modal_attention_ia.sum(dim=-1, keepdim=True))
        
        img_slot_out['cross_attn'] = cross_modal_attention_ai
        aud_slot_out['cross_attn'] = cross_modal_attention_ia
            
        loss_info_nce, loss_match, loss_div, loss_recon = compute_loss(img_slot_out, aud_slot_out, args, mode='train')

        loss = loss_info_nce + loss_match + loss_div + loss_recon
        if args.slots_maxsim == 'True' and args.use_perm_reg == 'True' and loss_perm is not None:
            loss = loss + args.lambda_perm * loss_perm      

        # Update running averages
        avg_total_loss.update(loss.item())
        avg_info_nce_loss.update(loss_info_nce.item())
        avg_match_loss.update(loss_match.item())
        avg_div_loss.update(loss_div.item()) 
        avg_recon_loss.update(loss_recon.item())

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        optimizer.zero_grad()
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss value {loss.item()} at step {i}")
            continue

        loss.backward()

        # Add gradient clipping 
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # Remove this line: scheduler.step()  # Step after each batch instead of each epoch

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            pbar.set_postfix({'TOTAL': f'{loss.item():.4f}', 'CONTR': f'{loss_info_nce.item():.4f}', 'MATCH': f'{loss_match.item():.4f}', 'DIV': f'{loss_div.item():.4f}', 'REC': f'{loss_recon.item():.4f}'})
            
            # Log to wandb
            if args.wandb == 'True':
                wandb.log({
                    # Batch-level current losses (logged per step)
                    'train/batch_total_loss': loss.item(),
                    'train/batch_info_nce_loss': loss_info_nce.item(),
                    'train/batch_matching_loss': loss_match.item(),
                    'train/batch_divergence_loss': loss_div.item(),
                    'train/batch_reconstruction_loss': loss_recon.item(),
                    **({'train/batch_perm_loss': loss_perm.item()} if (args.slots_maxsim == 'True' and args.use_perm_reg == 'True' and loss_perm is not None) else {}),
                    # Batch-level average losses (logged per step)
                    'train/batch_avg_total_loss': avg_total_loss.avg,
                    'train/batch_avg_info_nce_loss': avg_info_nce_loss.avg,
                    'train/batch_avg_matching_loss': avg_match_loss.avg,
                    'train/batch_avg_divergence_loss': avg_div_loss.avg,
                    'train/batch_avg_reconstruction_loss': avg_recon_loss.avg,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/batch_step': train_step + i  # Step for batch-level logging
                })

        del loss

    # Log epoch averages to wandb
    if args.wandb == 'True':
        wandb.log({
            'train/epoch_avg_total_loss': avg_total_loss.avg,
            'train/epoch_avg_info_nce_loss': avg_info_nce_loss.avg,
            'train/epoch_avg_matching_loss': avg_match_loss.avg,
            'train/epoch_avg_divergence_loss': avg_div_loss.avg,
            'train/epoch_avg_reconstruction_loss': avg_recon_loss.avg,
            'train/epoch': epoch,  # Same epoch number as validation
            'epoch': epoch  # Common epoch key for plotting train vs val together
        })


def log_file_visualizations(dataset, model, file_list, mode, epoch, args):
    """
    Log visualizations for specific files after epoch is complete.
    Model should be in eval mode when calling this function.
    
    Args:
        dataset: The dataset to get samples from (or the underlying dataset if using Subset)
        model: The model (will be set to eval mode)
        file_list: List of file IDs to log visualizations for
        mode: 'train' or 'val'
        epoch: Current epoch number
        args: Training arguments
    """
    if not args.wandb == 'True' or not file_list:
        return
    
    model.eval()
    
    # Handle Subset wrapper to get original dataset
    original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    # Find indices of files in the dataset
    file_indices = []
    for file_id in file_list:
        # Match file_id with dataset filenames
        for idx in range(len(original_dataset)):
            dataset_file = original_dataset.image_files[idx].split('.')[0]  # Remove .jpg extension
            if file_id in dataset_file or dataset_file == file_id:
                file_indices.append(idx)
                break
    
    if not file_indices:
        print(f"Warning: No matching files found in {mode} dataset for logging")
        return
    
    # Create a subset dataset or DataLoader for these specific indices
    from torch.utils.data import Subset
    # Use original_dataset to create subset to avoid nested Subset issues
    subset_dataset = Subset(original_dataset, file_indices)
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    with torch.no_grad():
        for batch_idx, (image, spec, bboxes, filename) in enumerate(subset_loader):
            if args.gpu is not None:
                spec = spec.cuda(args.gpu, non_blocking=True)
                image = image.cuda(args.gpu, non_blocking=True)
            
            B = image.shape[0]
            
            aud_slot_out, img_slot_out = model(image.float(), spec.float())
            
            if args.slots_maxsim == 'True':
                # Normalize per-slot vectors on channel dim
                aud_slots = F.normalize(aud_slot_out['slots'], dim=2)
                img_slots = F.normalize(img_slot_out['slots'], dim=2)
                
                # Pairwise similarity S[b, n_a, n_i]
                similarity_slots = torch.einsum('bnc,bmc->bnm', aud_slots, img_slots)

                B, N, C = aud_slots.shape
                assert N == 2, "STE sort implemented for num_slots=2"

                # Audio STE permutation
                a0_best = similarity_slots[:, 0, :].max(dim=1).values
                a1_best = similarity_slots[:, 1, :].max(dim=1).values
                logits_a = (a1_best - a0_best) / 0.3
                p_a = torch.sigmoid(logits_a)
                P_a_soft = torch.stack([
                    torch.stack([1 - p_a, p_a], dim=1),
                    torch.stack([p_a, 1 - p_a], dim=1)
                ], dim=1)
                a_choose1 = (p_a > 0.5).long()
                P_a_hard = torch.zeros_like(P_a_soft)
                P_a_hard[torch.arange(B, device=P_a_soft.device), 0, a_choose1] = 1
                P_a_hard[torch.arange(B, device=P_a_soft.device), 1, 1 - a_choose1] = 1
                P_a = P_a_hard.detach() - P_a_soft.detach() + P_a_soft

                # Image STE permutation
                i0_best = similarity_slots[:, :, 0].max(dim=1).values
                i1_best = similarity_slots[:, :, 1].max(dim=1).values
                logits_i = (i1_best - i0_best) / 0.3
                p_i = torch.sigmoid(logits_i)
                P_i_soft = torch.stack([
                    torch.stack([1 - p_i, p_i], dim=1),
                    torch.stack([p_i, 1 - p_i], dim=1)
                ], dim=1)
                i_choose1 = (p_i > 0.5).long()
                P_i_hard = torch.zeros_like(P_i_soft)
                P_i_hard[torch.arange(B, device=P_i_soft.device), 0, i_choose1] = 1
                P_i_hard[torch.arange(B, device=P_i_soft.device), 1, 1 - i_choose1] = 1
                P_i = P_i_hard.detach() - P_i_soft.detach() + P_i_soft

                aud_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['slots'])
                img_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['slots'])
                
                aud_slot_out['q_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['q'])
                img_slot_out['q_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['q'])
                
                aud_slot_out['attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['intra_attn'])
                img_slot_out['attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['intra_attn'])
                
                img_slot_out['debug_dots_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['debug_dots'].squeeze(1))
                aud_slot_out['debug_dots_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['debug_dots'].squeeze(1))
                
                img_slot_out['debug_attn_pre_norm_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['debug_attn_pre_norm'].squeeze(1))
                aud_slot_out['debug_attn_pre_norm_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['debug_attn_pre_norm'].squeeze(1))
                
                img_slot_out['debug_attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['debug_attn'].squeeze(1))
                aud_slot_out['debug_attn_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['debug_attn'].squeeze(1))

            else:
                aud_slot_out['slots_sorted'] = aud_slot_out['slots']
                img_slot_out['slots_sorted'] = img_slot_out['slots']

                aud_slot_out['q_sorted'] = aud_slot_out['q']
                img_slot_out['q_sorted'] = img_slot_out['q']
                
                aud_slot_out['attn_sorted'] = aud_slot_out['intra_attn']
                img_slot_out['attn_sorted'] = img_slot_out['intra_attn']
                
                img_slot_out['debug_dots_sorted'] = img_slot_out['debug_dots']
                aud_slot_out['debug_dots_sorted'] = aud_slot_out['debug_dots']
                
                img_slot_out['debug_attn_pre_norm_sorted'] = img_slot_out['debug_attn_pre_norm']
                aud_slot_out['debug_attn_pre_norm_sorted'] = aud_slot_out['debug_attn_pre_norm']
                
                img_slot_out['debug_attn_sorted'] = img_slot_out['debug_attn']
                aud_slot_out['debug_attn_sorted'] = aud_slot_out['debug_attn']
            
            # Compute cross-modal attention
            cross_modal_attention_ai = torch.einsum('bid,bjd->bij', aud_slot_out['q_sorted'], img_slot_out['k']) * (512 ** -0.5)
            cross_modal_attention_ai = cross_modal_attention_ai.softmax(dim=1) + 1e-8
            h = w = int(cross_modal_attention_ai.shape[-1] ** 0.5)
            cross_modal_attention_ai = (cross_modal_attention_ai / cross_modal_attention_ai.sum(dim=-1, keepdim=True))

            cross_modal_attention_ia = torch.einsum('bid,bjd->bij', img_slot_out['q_sorted'], aud_slot_out['k']) * (512 ** -0.5)
            cross_modal_attention_ia = cross_modal_attention_ia.softmax(dim=1) + 1e-8
            cross_modal_attention_ia = (cross_modal_attention_ia / cross_modal_attention_ia.sum(dim=-1, keepdim=True))
            
            img_slot_out['cross_attn'] = cross_modal_attention_ai
            aud_slot_out['cross_attn'] = cross_modal_attention_ia
            
            # Prepare attention maps for visualization
            cross_modal_attention_image = img_slot_out['cross_attn'].contiguous().view(B, 2, h, w)
            cross_modal_attention_image = F.interpolate(cross_modal_attention_image, size=(224, 224), mode='bilinear', align_corners=False).data.cpu().numpy()
            
            cross_modal_attention_audio = aud_slot_out['cross_attn'].contiguous().view(B, 2, 7)
            cross_modal_attention_audio = F.interpolate(cross_modal_attention_audio, size=200, mode='linear', align_corners=False).unsqueeze(2).repeat(1, 1, 257, 1).data.cpu().numpy()  # (B, 2, 257, 200)
            
            # Intra-modal Attention
            intra_modal_attention_image = img_slot_out['attn_sorted'].contiguous().view(B, 2, h, w)
            intra_modal_attention_image = F.interpolate(intra_modal_attention_image, size=(224, 224), mode='bilinear', align_corners=False).data.cpu().numpy()

            intra_modal_attention_audio = aud_slot_out['attn_sorted'].contiguous().view(B, 2, 7)
            intra_modal_attention_audio = F.interpolate(intra_modal_attention_audio, size=200, mode='linear', align_corners=False).unsqueeze(2).repeat(1, 1, 257, 1).data.cpu().numpy()  # (B, 2, 257, 200)
            
            # Similarity Embeddings
            img_emb = F.normalize(img_slot_out['embedding_original'].contiguous().view(B, 512, h, w), dim=1)
            aud_emb = F.normalize(aud_slot_out['embedding_original'], dim=1)

            similarity_embeddings = torch.einsum('bihw,bi->bhw', img_emb, aud_emb)
            similarity_embeddings = F.interpolate(similarity_embeddings.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).data.cpu().numpy()
            
            # Also keep original-resolution (7x7) attentions for image/audio to log overlays at "attention-native" size
            img_intra_attn_7 = img_slot_out['attn_sorted'].contiguous().view(B, 2, h, w).data.cpu().numpy()  # (B,2,7,7)
            img_cross_attn_7 = img_slot_out['cross_attn'].contiguous().view(B, 2, h, w).data.cpu().numpy()  # (B,2,7,7)
            aud_intra_attn_7 = aud_slot_out['attn_sorted'].contiguous().view(B, 2, 7, 1).repeat(1, 1, 1, 7).data.cpu().numpy()  # (B,2,7,7)
            aud_cross_attn_7 = aud_slot_out['cross_attn'].contiguous().view(B, 2, 7, 1).repeat(1, 1, 1, 7).data.cpu().numpy()  # (B,2,7,7)

            import cv2

            # Log visualizations for each sample
            for i in range(B):
                # Get original image and spectrogram
                orig_img = inverse_normalize(image[i]).cpu().permute(1,2,0).numpy()
                orig_img = np.clip(orig_img, 0, 1)  # Clip to valid range
                
                orig_spec = spec[i,0].data.cpu().numpy()

                # Image logs
                similarity_embeddings_target = utils.normalize_img(similarity_embeddings[i, 0])
                
                fig_similarity = gen_pred_figure(orig_img, similarity_embeddings_target)
                wandb.log({f"{filename[i]}_{mode}/similarity": wandb.Image(fig_similarity), 'epoch': epoch})
                if args.debug =='True':
                    dir_path = os.path.join('debug_logs', f'img_sim', mode, f'{filename[i]}')
                    os.makedirs(dir_path, exist_ok=True)
                    imsim_path = os.path.join(dir_path, f'ep_{epoch}.jpg')
                    plt.savefig(imsim_path)
                plt.close(fig_similarity)
                
                intra_modal_image_target = utils.normalize_img(intra_modal_attention_image[i, 0])
                intra_modal_image_offtarget = utils.normalize_img(intra_modal_attention_image[i, 1])
                
                fig_intramodal = gen_pred_figure(orig_img, intra_modal_image_target, intra_modal_image_offtarget)
                wandb.log({f"{filename[i]}_{mode}/intramodal_image": wandb.Image(fig_intramodal), 'epoch': epoch})
                if args.debug =='True':
                    dir_path = os.path.join('debug_logs', f'img_intramodal', mode, f'{filename[i]}')
                    os.makedirs(dir_path, exist_ok=True)
                    im_intramodal_path = os.path.join(dir_path, f'ep_{epoch}.jpg')
                    plt.savefig(im_intramodal_path)
                plt.close(fig_intramodal)
                
                crossmodal_image_target = utils.normalize_img(cross_modal_attention_image[i, 0])
                crossmodal_image_offtarget = utils.normalize_img(cross_modal_attention_image[i, 1])
                
                fig_crossmodal_image = gen_pred_figure(orig_img, crossmodal_image_target, crossmodal_image_offtarget)
                wandb.log({f"{filename[i]}_{mode}/crossmodal_image": wandb.Image(fig_crossmodal_image), 'epoch': epoch})
                if args.debug =='True':
                    dir_path = os.path.join('debug_logs', f'img_crossmodal', mode, f'{filename[i]}')
                    os.makedirs(dir_path, exist_ok=True)
                    im_crossmodal_path = os.path.join(dir_path, f'ep_{epoch}.jpg')
                    plt.savefig(im_crossmodal_path)
                plt.close(fig_crossmodal_image)
                
                # Audio logs
                fig_spectrogram = plt.figure(figsize=(10, 5))
                ax = fig_spectrogram.add_subplot(111)
                ax.imshow(orig_spec, origin='lower', aspect='auto', cmap='magma')
                ax.set_title('Spectrogram')
                ax.axis('off')
                wandb.log({f"{filename[i]}_{mode}/spectrogram": wandb.Image(fig_spectrogram), 'epoch': epoch})
                if args.debug =='True':
                    dir_path = os.path.join('debug_logs', f'spec', mode, f'{filename[i]}')
                    os.makedirs(dir_path, exist_ok=True)
                    spectrogram_path = os.path.join(dir_path, f'ep_{epoch}.jpg')
                    plt.savefig(spectrogram_path)
                plt.close(fig_spectrogram)
                
                intra_modal_audio_target = utils.normalize_img(intra_modal_attention_audio[i, 0])
                intra_modal_audio_offtarget = utils.normalize_img(intra_modal_attention_audio[i, 1])
                
                fig_intramodal_audio = gen_spec_pred_figure(orig_spec, intra_modal_audio_target, intra_modal_audio_offtarget)
                wandb.log({f"{filename[i]}_{mode}/intramodal_audio": wandb.Image(fig_intramodal_audio), 'epoch': epoch})
                if args.debug =='True':
                    dir_path = os.path.join('debug_logs', f'aud_intramodal', mode, f'{filename[i]}')
                    os.makedirs(dir_path, exist_ok=True)
                    aud_intramodal_path = os.path.join(dir_path, f'ep_{epoch}.jpg')
                    plt.savefig(aud_intramodal_path)
                plt.close(fig_intramodal_audio)
                
                crossmodal_audio_target = utils.normalize_img(cross_modal_attention_audio[i, 0])
                crossmodal_audio_offtarget = utils.normalize_img(cross_modal_attention_audio[i, 1])
                
                fig_crossmodal_audio = gen_spec_pred_figure(orig_spec, crossmodal_audio_target, crossmodal_audio_offtarget)
                wandb.log({f"{filename[i]}_{mode}/crossmodal_audio": wandb.Image(fig_crossmodal_audio), 'epoch': epoch})
                if args.debug =='True':
                    dir_path =  os.path.join('debug_logs', f'aud_crossmodal', mode, f'{filename[i]}')
                    os.makedirs(dir_path, exist_ok=True)
                    aud_crossmodal_path = os.path.join(dir_path, f'ep_{epoch}.jpg')
                    plt.savefig(aud_crossmodal_path)
                plt.close(fig_crossmodal_audio)

                # ============================================================
                # NEW: "native attention size" logs
                # - Image resized to 7x7 for image-attn overlays
                # - Spectrogram resized to 7x7 for audio-attn overlays
                # ============================================================
                img7 = cv2.resize(orig_img, (7, 7), interpolation=cv2.INTER_AREA)
                img7_up = cv2.resize(img7, (224, 224), interpolation=cv2.INTER_NEAREST)  # for display

                spec7 = cv2.resize(orig_spec, (7, 7), interpolation=cv2.INTER_AREA)
                spec7_up = cv2.resize(spec7, (257, 200), interpolation=cv2.INTER_NEAREST)  # for display

                # Image-side native overlays (use 7x7 attention, upscale to 224 for the helper)
                img_intra7_t = utils.normalize_img(img_intra_attn_7[i, 0])
                img_intra7_o = utils.normalize_img(img_intra_attn_7[i, 1])
                img_cross7_t = utils.normalize_img(img_cross_attn_7[i, 0])
                img_cross7_o = utils.normalize_img(img_cross_attn_7[i, 1])

                img_intra7_t_up = cv2.resize(img_intra7_t, (224, 224), interpolation=cv2.INTER_NEAREST)
                img_intra7_o_up = cv2.resize(img_intra7_o, (224, 224), interpolation=cv2.INTER_NEAREST)
                img_cross7_t_up = cv2.resize(img_cross7_t, (224, 224), interpolation=cv2.INTER_NEAREST)
                img_cross7_o_up = cv2.resize(img_cross7_o, (224, 224), interpolation=cv2.INTER_NEAREST)

                fig_img7_intra = gen_pred_figure(img7_up, img_intra7_t_up, img_intra7_o_up)
                wandb.log({f"{filename[i]}_{mode}/intramodal_image_native7x7": wandb.Image(fig_img7_intra), 'epoch': epoch})
                plt.close(fig_img7_intra)

                fig_img7_cross = gen_pred_figure(img7_up, img_cross7_t_up, img_cross7_o_up)
                wandb.log({f"{filename[i]}_{mode}/crossmodal_image_native7x7": wandb.Image(fig_img7_cross), 'epoch': epoch})
                plt.close(fig_img7_cross)
                
                # Log debug values from slot attention iterations
                if args.wandb == 'True' and args.log_debug_attentions == 'True':
                    # Log image slot attention debug values
                    if 'debug_dots_sorted' in img_slot_out:
                        for it_idx, dots in enumerate(img_slot_out['debug_dots_sorted']):
                            dots_np = dots[0].cpu().numpy().reshape(dots[0].shape[0], 7, 7)  # (num_slots, 7, 7)
                            fig = gen_debug_slot_figure(dots_np, f'Image Dots - Iteration {it_idx}', modality='image')
                            wandb.log({f'{filename[i]}_{mode}/debug/dots_img_it{it_idx}': wandb.Image(fig), 'epoch': epoch})
                            plt.close(fig)
                    
                    if 'debug_attn_pre_norm_sorted' in img_slot_out:
                        for it_idx, attn_pre_norm in enumerate(img_slot_out['debug_attn_pre_norm_sorted']):
                            attn_pre_norm_np = attn_pre_norm[0].cpu().numpy().reshape(attn_pre_norm[0].shape[0], 7, 7)  # (num_slots, 7, 7)
                            fig = gen_debug_slot_figure(attn_pre_norm_np, f'Image Attn Pre Norm - Iteration {it_idx}', modality='image')
                            wandb.log({f'{filename[i]}_{mode}/debug/attn_pre_norm_img_it{it_idx}': wandb.Image(fig), 'epoch': epoch})
                            plt.close(fig)
                    
                    if 'debug_attn_sorted' in img_slot_out:
                        for it_idx, attn in enumerate(img_slot_out['debug_attn_sorted']):
                            attn_np = attn[0].cpu().numpy().reshape(attn[0].shape[0], 7, 7)  # (num_slots, 7, 7)
                            fig = gen_debug_slot_figure(attn_np, f'Image Attn - Iteration {it_idx}', modality='image')
                            wandb.log({f'{filename[i]}_{mode}/debug/attn_img_it{it_idx}': wandb.Image(fig), 'epoch': epoch})
                            plt.close(fig)
                    
                    # Log audio slot attention debug values
                    if 'debug_dots_sorted' in aud_slot_out:
                        for it_idx, dots in enumerate(aud_slot_out['debug_dots_sorted']):
                            dots_np = dots[0].cpu().numpy()  # (num_slots, seq_len)
                            fig = gen_debug_slot_figure(dots_np, f'Audio Dots - Iteration {it_idx}', modality='audio')
                            wandb.log({f'{filename[i]}_{mode}/debug/dots_aud_it{it_idx}': wandb.Image(fig), 'epoch': epoch})
                            plt.close(fig)
                    
                    if 'debug_attn_pre_norm_sorted' in aud_slot_out:
                        for it_idx, attn_pre_norm in enumerate(aud_slot_out['debug_attn_pre_norm_sorted']):
                            attn_pre_norm_np = attn_pre_norm[i].cpu().numpy()  # (num_slots, seq_len)
                            fig = gen_debug_slot_figure(attn_pre_norm_np, f'Audio Attn Pre Norm - Iteration {it_idx}', modality='audio')
                            wandb.log({f'{filename[0]}_{mode}/debug/attn_pre_norm_aud_it{it_idx}': wandb.Image(fig), 'epoch': epoch})
                            plt.close(fig)
                    
                    if 'debug_attn_sorted' in aud_slot_out:
                        for it_idx, attn in enumerate(aud_slot_out['debug_attn_sorted']):
                            attn_np = attn[i].cpu().numpy()  # (num_slots, seq_len)
                            fig = gen_debug_slot_figure(attn_np, f'Audio Attn - Iteration {it_idx}', modality='audio')
                            wandb.log({f'{filename[i]}_{mode}/debug/attn_aud_it{it_idx}': wandb.Image(fig), 'epoch': epoch})
                            plt.close(fig)


def validate(test_loader, model, args, epoch):
    model.train(False)
    avg_loss = AverageMeter('Validation Loss', ':.4f')
    
    with torch.no_grad():
        for step, (image, spec, bboxes, filename) in enumerate(tqdm.tqdm(test_loader, desc='Validating')):
            
            # Add training step counter
            val_step = epoch * len(test_loader) + step  # Start from epoch beginning
            if args.gpu is not None:
                spec = spec.cuda(args.gpu, non_blocking=True)
                image = image.cuda(args.gpu, non_blocking=True)
            
            B = image.shape[0]
            
            aud_slot_out, img_slot_out = model(image.float(), spec.float())

            if args.slots_maxsim == 'True':
                # Normalize per-slot vectors over channel dim
                aud_slots = F.normalize(aud_slot_out['slots'], dim=2)
                img_slots = F.normalize(img_slot_out['slots'], dim=2)
                
                # Pairwise similarities
                similarity_slots = torch.einsum('bnc,bmc->bnm', aud_slots, img_slots)

                B, N, C = aud_slots.shape
                assert N == 2, "STE sort implemented for num_slots=2"

                # Audio STE permutation
                a0_best = similarity_slots[:, 0, :].max(dim=1).values
                a1_best = similarity_slots[:, 1, :].max(dim=1).values
                logits_a = (a1_best - a0_best) / 0.3
                p_a = torch.sigmoid(logits_a)
                P_a_soft = torch.stack([
                    torch.stack([1 - p_a, p_a], dim=1),
                    torch.stack([p_a, 1 - p_a], dim=1)
                ], dim=1)
                a_choose1 = (p_a > 0.5).long()
                P_a_hard = torch.zeros_like(P_a_soft)
                P_a_hard[torch.arange(B, device=P_a_soft.device), 0, a_choose1] = 1
                P_a_hard[torch.arange(B, device=P_a_soft.device), 1, 1 - a_choose1] = 1
                P_a = P_a_hard.detach() - P_a_soft.detach() + P_a_soft

                # Image STE permutation
                i0_best = similarity_slots[:, :, 0].max(dim=1).values
                i1_best = similarity_slots[:, :, 1].max(dim=1).values
                logits_i = (i1_best - i0_best) / 0.3
                p_i = torch.sigmoid(logits_i)
                P_i_soft = torch.stack([
                    torch.stack([1 - p_i, p_i], dim=1),
                    torch.stack([p_i, 1 - p_i], dim=1)
                ], dim=1)
                i_choose1 = (p_i > 0.5).long()
                P_i_hard = torch.zeros_like(P_i_soft)
                P_i_hard[torch.arange(B, device=P_i_soft.device), 0, i_choose1] = 1
                P_i_hard[torch.arange(B, device=P_i_soft.device), 1, 1 - i_choose1] = 1
                P_i = P_i_hard.detach() - P_i_soft.detach() + P_i_soft

                aud_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_a, aud_slot_out['slots'])
                img_slot_out['slots_sorted'] = torch.einsum('bnm,bmc->bnc', P_i, img_slot_out['slots'])
                
            else:
                aud_slot_out['slots_sorted'] = aud_slot_out['slots']
                img_slot_out['slots_sorted'] = img_slot_out['slots']
            
            loss_info_nce = compute_loss(img_slot_out, aud_slot_out, args, mode='val')
            avg_loss.update(loss_info_nce.item())

            # # Cross-modal Attention
            # cross_modal_attention_ai = torch.einsum('bid,bjd->bij', aud_slot_out['q_sorted'], img_slot_out['k']) * (512 ** -0.5)
            # cross_modal_attention_ai = cross_modal_attention_ai.softmax(dim=1) + 1e-8
            # h = w = int(cross_modal_attention_ai.shape[-1] ** 0.5)
            # cross_modal_attention_ai = (cross_modal_attention_ai / cross_modal_attention_ai.sum(dim=-1, keepdim=True)).contiguous().view(B, 2, h, w)
            # cross_modal_attention_ai = F.interpolate(cross_modal_attention_ai, size=(224, 224), mode='bilinear', align_corners=False).cpu().numpy()
            
            # cross_modal_attention_ia = torch.einsum('bid,bjd->bij', img_slot_out['q_sorted'], aud_slot_out['k']) * (512 ** -0.5)
            # cross_modal_attention_ia = cross_modal_attention_ia.softmax(dim=1) + 1e-8
            # cross_modal_attention_ia = (cross_modal_attention_ia / cross_modal_attention_ia.sum(dim=-1, keepdim=True)).contiguous()
            # cross_modal_attention_ia = F.interpolate(cross_modal_attention_ia, size=200, mode='linear', align_corners=False).unsqueeze(2).repeat(1, 1, 257, 1).data.cpu().numpy()  # (B, 2, 257, 200)
            
            # # Intra-modal Attention
            # intra_modal_attention_i = img_slot_out['attn_sorted'].contiguous().view(B, 2, h, w)
            # intra_modal_attention_i = F.interpolate(intra_modal_attention_i, size=(224, 224), mode='bilinear', align_corners=False).cpu().numpy()
            
            # intra_modal_attention_a = aud_slot_out['attn_sorted'].contiguous().view(B, 2, 7)
            # intra_modal_attention_a = F.interpolate(intra_modal_attention_a, size=200, mode='linear', align_corners=False).unsqueeze(2).repeat(1, 1, 257, 1).data.cpu().numpy()  # (B, 2, 257, 200)
            
            # # Similarity Embeddings
            # img_emb = F.normalize(img_slot_out['embedding_original'].contiguous().view(B, 512, h, w), dim=1)
            # aud_emb = F.normalize(aud_slot_out['embedding_original'], dim=1)

            # similarity_embeddings = torch.einsum('bihw,bi->bhw', img_emb, aud_emb)
            # similarity_embeddings = F.interpolate(similarity_embeddings.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).cpu().numpy()
                        
            # Log current loss
            if args.wandb == 'True':
                wandb.log({
                    'val/batch_current_loss': loss_info_nce.item(),
                    'val/batch_avg_loss': avg_loss.avg,  # Batch-level running average
                    'val/batch_step': val_step,  # Step for batch-level logging
                })

        # Log epoch-level validation metrics
        if args.wandb == 'True':
            wandb.log({
                'val/epoch_avg_loss': avg_loss.avg,  # Epoch-level average loss
                'val/epoch_info_nce_loss': avg_loss.avg,  # Alias for consistency
                'val/epoch': epoch,  # Same epoch number as training
                'epoch': epoch  # Common epoch key for plotting train vs val together
            })
    return loss_info_nce.item()


def validate_vggss(test_loader, model, args, epoch):
    model.train(False)
    evaluator = utils.Evaluator()
    for step, (image, spec, bboxes, _) in enumerate(tqdm.tqdm(test_loader, desc='Validating')):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        aud_slot_out, img_slot_out = model(image.float(), spec.float())

        # loss_info_nce = compute_loss(img_slot_out, aud_slot_out, args, mode='val')

        avl_map = img_slot_out['cross_attn'].reshape(1, 2, 7, 7)

        avl_map = F.interpolate(avl_map, size=(224, 224), mode='bicubic', align_corners=False)
        avl_map = avl_map.data.cpu().numpy()

        for i in range(spec.shape[0]):
            # Get prediction and ground truth
            pred = utils.normalize_img(avl_map[i, 0])
            off_target = utils.normalize_img(avl_map[i, 1])

            gt_map = bboxes['gt_map'].data.cpu().numpy()
            
            # Create visualization and log to wandb
            if args.wandb == 'True':
                if step < 2:  # Only for first 2 elements of validation epoch
                    # Inverse normalize using ImageNet mean/std
                    orig_img = inverse_normalize(image[i]).cpu().permute(1,2,0).numpy()
                    orig_img = np.clip(orig_img, 0, 1)  # Clip to valid range
                    
                    fig = create_visualization(orig_img, pred, off_target)
                    wandb.log({f"val/pred_overlay_{step}_{i}": wandb.Image(fig)})
                    plt.close(fig)

            thr = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            evaluator.cal_CIOU(pred, gt_map, thr)

    cIoU = evaluator.finalize_AP50()
    AUC = evaluator.finalize_AUC()

    # Log validation metrics to wandb (only on rank 0)
    if args.wandb == 'True' and args.rank == 0:
        wandb.log({
            'val/cIoU': cIoU,
            'val/AUC': AUC,
            'val/epoch': epoch
        })
    return cIoU, AUC


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def create_visualization(image, pred, off_target, gt_map=None):
    """Create visualization with original image, prediction overlay and bounding box
    
    Args:
        image: Original image as numpy array (H,W,3)
        pred: Prediction heatmap as numpy array (H,W) 
        gt_map: Ground truth binary map as numpy array (H,W)
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left subplot - target prediction
    ax1.imshow(image)
    ax1.imshow(pred, alpha=0.5, cmap='jet')
    ax1.axis('off')
    
    # Right subplot - off-target prediction
    ax2.imshow(image) 
    ax2.imshow(off_target, alpha=0.5, cmap='jet')
    ax2.axis('off')
    
    # Draw ground truth bounding box on both plots
    if gt_map is not None:
        y_idx, x_idx = np.where(gt_map == 1)
        if len(y_idx) > 0:
            min_x, max_x = x_idx.min(), x_idx.max()
            min_y, max_y = y_idx.min(), y_idx.max()
            
            rect1 = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                linewidth=2, edgecolor='g', facecolor='none')
            rect2 = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                linewidth=2, edgecolor='g', facecolor='none')
            ax1.add_patch(rect1)
            ax2.add_patch(rect2)
    
    ax1.set_title('Target Prediction')
    ax2.set_title('Off-Target Prediction')
    
    plt.tight_layout()
    return fig

def gen_spec_pred_figure(orig_spec, pred=None, off_target=None):
    """
    Create a visualization with 3 subfigures (vertically stacked):
      - 1st: original spectrogram
      - 2nd: target overlayed on spectrogram
      - 3rd: off-target overlayed on spectrogram

    Args:
        orig_spec (ndarray): Original spectrogram (H,W,3 or H,W)
        pred (ndarray): Prediction heatmap (H,W)
        off_target (ndarray): Off-target heatmap (H,W)
    Returns:
        fig (matplotlib.figure.Figure): The resulting Figure object
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(24, 10))

    # Row 1: original spectrogram
    axs[0].imshow(orig_spec, origin='lower', aspect='auto', cmap='magma')
    axs[0].set_title('Spectrogram')
    axs[0].axis('off')

    # Row 2: target overlay
    axs[1].imshow(orig_spec, origin='lower', aspect='auto', cmap='magma')
    if pred is not None:
        axs[1].imshow(pred, cmap='jet', alpha=0.8, aspect='auto', origin='lower')
    axs[1].set_title('Target Prediction Overlay')
    axs[1].axis('off')

    # Row 3: off-target overlay
    axs[2].imshow(orig_spec, origin='lower', aspect='auto', cmap='magma')
    if off_target is not None:
        axs[2].imshow(off_target, cmap='jet', alpha=0.8, aspect='auto', origin='lower')
    axs[2].set_title('Off-Target Overlay')
    axs[2].axis('off')

    plt.tight_layout()
    return fig


def gen_debug_slot_figure(data_np, title, modality='image'):
    """
    Create a figure with subplots showing each slot's attention/dots.
    
    Args:
        data_np: numpy array of shape (num_slots, ...) 
                 - For images: (num_slots, 7, 7)
                 - For audio: (num_slots, seq_len)
        title: Title for the figure
        modality: 'image' or 'audio' to determine labels
    
    Returns:
        matplotlib figure
    """
    num_slots = data_np.shape[0]
    fig, axes = plt.subplots(1, num_slots, figsize=(6 * num_slots, 5))
    
    # Handle single subplot case
    if num_slots == 1:
        axes = [axes]
    
    for slot_idx in range(num_slots):
        if modality == 'image' and len(data_np.shape) == 3:
            # Image case: (num_slots, 7, 7)
            im = axes[slot_idx].imshow(data_np[slot_idx], aspect='auto', cmap='jet')
            axes[slot_idx].set_xlabel('X Position')
            axes[slot_idx].set_ylabel('Y Position')
        else:
            # Audio case: (num_slots, seq_len)
            # Reshape 1D array to 2D for visualization (1, seq_len) - horizontal temporal dimension
            slot_data = data_np[slot_idx]
            if len(slot_data.shape) == 1:
                slot_data = slot_data.reshape(1, -1)  # (1, 7) - horizontal layout
            im = axes[slot_idx].imshow(slot_data, aspect='auto', cmap='jet')
            axes[slot_idx].set_xlabel('Sequence Position')
            axes[slot_idx].set_ylabel('')
        
        axes[slot_idx].set_title(f'Slot {slot_idx}')
        plt.colorbar(im, ax=axes[slot_idx])
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig

def gen_pred_figure(orig_img, pred=None, off_target=None):
    if off_target is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(orig_img)
        axs[0].imshow(pred, cmap='jet', alpha=0.5)
        axs[0].set_title('Prediction')
        axs[0].axis('off')

        axs[1].imshow(orig_img)
        axs[1].imshow(off_target, cmap='jet', alpha=0.5)
        axs[1].set_title('Off-Target')
        axs[1].axis('off')

        plt.tight_layout()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.imshow(orig_img)
        axs.imshow(pred, cmap='jet', alpha=0.5)
        axs.set_title('Prediction')
        axs.axis('off')
    return fig
            
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())

