import os
import matplotlib.pyplot as plt
from matplotlib import patches
import tqdm
import argparse
import builtins
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist
import wandb

import utils
from model import EZVSL
from losses import compute_loss
from datasets import get_train_dataset, get_test_dataset, get_train_test_dataset


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
    parser.add_argument('--num_slots', default=2, type=int)
    parser.add_argument('--iters', default=5, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--slots_no_W', default='False', type=str)

    parser.add_argument('--lambda_info_nce', default=1.0, type=float)
    parser.add_argument('--lambda_match', default=100.0, type=float)
    parser.add_argument('--lambda_div', default=0.1, type=float)
    parser.add_argument('--lambda_recon', default=0.1, type=float)

    # training/evaluation parameters
    parser.add_argument('--debug', type=str, default='True', help='debug mode')
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
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
    optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)

    # Resume if possible
    start_epoch, best_cIoU, best_Auc = 0, 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cpu')
        start_epoch, best_cIoU, best_Auc = ckp['epoch'], ckp['best_cIoU'], ckp['best_Auc']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    # Dataloaders
    if args.testset == 'vggss_144k':
        traindataset, testdataset = get_train_test_dataset(args)
    else:
        traindataset = get_train_dataset(args)
        testdataset = get_test_dataset(args)

    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
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
        
    best_loss_info_nce = loss_info_nce
    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, epoch, args)

        # Evaluate
        if args.testset == 'vggss_144k':
            loss_info_nce = validate(test_loader, model, args, start_epoch)
            print(f'    ----Validation epoch {start_epoch}----')
            print(f'    Info NCE Loss (epoch {start_epoch}): {loss_info_nce:.4f}')
        else:
            cIoU, auc = validate_vggss(test_loader, model, args, start_epoch)
            print(f'    ----Validation epoch {start_epoch}----')
            print(f'    cIoU (epoch {start_epoch}): {cIoU:.4f}')
            print(f'    AUC (epoch {start_epoch}): {auc:.4f}')

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
        if args.rank == 0:
            ckp = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1,
                   'best_cIoU': best_cIoU,
                   'best_Auc': best_Auc}
            torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
            print(f"Model saved to {model_dir}")
        
        if args.testset == 'vggss_144k':
            if loss_info_nce >= best_loss_info_nce:
                best_loss_info_nce = loss_info_nce.item()
                if args.rank == 0:
                    torch.save(ckp, os.path.join(model_dir, 'best.pth'))
        else:
            if cIoU >= best_cIoU:
                best_cIoU, best_Auc = cIoU, auc
                if args.rank == 0:
                    torch.save(ckp, os.path.join(model_dir, 'best.pth'))
    
    # Finish wandb run
    if args.wandb == 'True' and args.rank == 0:
        wandb.finish()

def train(train_loader, model, optimizer, epoch, args):
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
    for i, (image, spec, _, _) in enumerate(pbar):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        aud_slot_out, img_slot_out = model(image.float(), spec.float())

        loss_info_nce, loss_match, loss_div, loss_recon = compute_loss(img_slot_out, aud_slot_out, args, mode='train')

        loss = args.lambda_info_nce * loss_info_nce + args.lambda_match * loss_match + args.lambda_div * loss_div + args.lambda_recon * loss_recon      

        # Update running averages
        avg_total_loss.update(loss.item())
        avg_info_nce_loss.update(loss_info_nce.item())
        avg_match_loss.update(loss_match.item())
        avg_div_loss.update(loss_div.item()) 
        avg_recon_loss.update(loss_recon.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            pbar.set_postfix({'TOTAL': f'{loss.item():.4f}', 'CONTR': f'{loss_info_nce.item():.4f}', 'MATCH': f'{loss_match.item():.4f}', 'DIV': f'{loss_div.item():.4f}', 'REC': f'{loss_recon.item():.4f}'})
            
            # Log to wandb (only on rank 0)
            if args.wandb == 'True':
                wandb.log({
                    'train/total_loss': loss.item(),
                    'train/info_nce_loss': loss_info_nce.item(),
                    'train/matching_loss': loss_match.item(),
                    'train/divergence_loss': loss_div.item(),
                    'train/reconstruction_loss': loss_recon.item(),
                    'train/epoch': epoch,
                    'train/step': epoch * len(train_loader) + i
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
            'train/epoch': epoch
        })


def validate(test_loader, model, args, epoch):
    model.train(False)
    avg_loss = AverageMeter('Validation Loss', ':.4f')
    
    for step, (image, spec, bboxes, _) in enumerate(tqdm.tqdm(test_loader, desc='Validating')):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
        
        B = image.shape[0]
        
        aud_slot_out, img_slot_out = model(image.float(), spec.float())

        loss_info_nce = compute_loss(img_slot_out, aud_slot_out, args, mode='val')
        avg_loss.update(loss_info_nce.item())

        avl_map = img_slot_out['cross_attn'].contiguous().view(B, 2, 7, 7)

        avl_map = F.interpolate(avl_map, size=(224, 224), mode='bicubic', align_corners=False)
        avl_map = avl_map.data.cpu().numpy()

        if args.wandb == 'True' and step == 0:
            for i in range(2):
                # Get prediction and ground truth
                pred = utils.normalize_img(avl_map[i, 0])
                off_target = utils.normalize_img(avl_map[i, 1])
            
                # Create visualization and log to wandb
                # Inverse normalize using ImageNet mean/std
                orig_img = inverse_normalize(image[i]).cpu().permute(1,2,0).numpy()
                orig_img = np.clip(orig_img, 0, 1)  # Clip to valid range
                
                fig = create_visualization(orig_img, pred, off_target)
                wandb.log({f"val/pred_overlay_{step}_{i}": wandb.Image(fig)})
                plt.close(fig)

        # Log current loss
        if args.wandb == 'True':
            wandb.log({
                'val/current_loss': loss_info_nce.item(),
                'val/step': step
            })

    # Log validation metrics to wandb (only on rank 0)
    if args.wandb == 'True' and args.rank == 0:
        wandb.log({
            'val/Info NCE Loss': loss_info_nce.item(),
            'val/avg_loss': avg_loss.avg,
            'val/epoch': epoch
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

        avl_map = img_slot_out['cross_attn'].contiguous().view(1, 2, 7, 7)

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
                    
                    fig = create_visualization(orig_img, pred, off_target, gt_map[i])
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

