import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, DistributedSampler

from tool.utils import init_distributed, init_seeds, is_main_process, cleanup_distributed
from tool.utils import mkdir_or_exist, load_config, save_ckpts, save_mask
from tool.utils import MyWcploss, focal_loss, SoftDiceLoss

from tool.logger import get_root_logger
from tool.sbudata import SBUCustomDataset
from tool.istddata import ISTDCustomDataset
from CLIP.clip import create_model
from model.shadowdino import ShadowDINO
from dinov3.backbones import dinov3_vitl16



def main(seed, args):
    cfg = load_config(f"{args.dataset}.yaml")
    mkdir_or_exist(cfg['ckpt_model'])
    mkdir_or_exist(cfg['ckpt_image'])
    if is_main_process(args):
        logger = get_root_logger(name='swin', log_file=cfg['logfile'], log_level=logging.INFO)
        logger.info(seed)
    t0 = time.perf_counter()
    last_log_t = t0


    image_size = cfg['dinov3']['img_size']        
    if args.dataset == 'sbu':
        train_dataset = SBUCustomDataset(image_size)
    else:
        train_dataset = ISTDCustomDataset(image_size)

    # DDP sampler
    batch_size_per_gpu = cfg['batch_size'] // args.world_size
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True,)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size_per_gpu, shuffle=False, drop_last=True, num_workers=8, pin_memory=True, sampler=train_sampler)

    Dinov3_model_path = './dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    Dino_model = dinov3_vitl16()
    state_dict = torch.load(Dinov3_model_path, map_location='cpu', weights_only=True)
    rdict = Dino_model.load_state_dict(state_dict, strict=True)
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device="cuda", pretrained='openai', require_pretrained=True)
    model = ShadowDINO(clip_model, Dino_model, device="cuda")
    model.cuda().train()

    # freeze clip & dinov3
    for p in model.clip_model.parameters():
        p.requires_grad = False
    for p in model.dinov3.parameters():
        p.requires_grad = False
    model.clip_model.eval()
    model.dinov3.eval()
    model = DDP(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=False)

    if is_main_process(args):
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params/1e6:.2f} M")
        print(f"Trainable params: {trainable/1e6:.2f} M")

    '''  4. loss && optimizer'''
    if cfg['loss']=='MyWcploss':
        mywbce = MyWcploss().cuda()
        dice_loss = SoftDiceLoss().cuda()
    if cfg['optimizer']['use']=='SGD':
        optimizer = torch.optim.SGD([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * 5e-3},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': 5e-3, \
                'weight_decay': cfg['optimizer']['weight_decay']}], momentum=cfg['optimizer']['momentum']
            )
    elif cfg['optimizer']['use']=='Adam':
        optimizer = torch.optim.Adam([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * 5e-3},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': 5e-3}], \
                betas=(0.9, 0.999), eps=1e-8
            )
    elif cfg['optimizer']['use']=='AdamW':
        optimizer = torch.optim.AdamW([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * 5e-3},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': 5e-3}], \
                betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg['optimizer']['weight_decay'])
 


    # ===== trainable params summary =====
    if is_main_process(args):
        print("=====")
        print("[Trainable parameters by name]")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"{n}: shape={tuple(p.shape)}, numel={p.numel()}")
        trainable_count = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {trainable_count} ({trainable_count/1e6:.2f} M)")
        print("=====")
    # ===== end summary =====


    '''  5. train  '''
    step=0
    train_loader_iter = iter(train_loader)
    if is_main_process(args):
        logger.info("start training")

    # sampler在epoch之间需要set_epoch()打乱。我们虽然是step循环，但dataloader会反复跑很多轮。
    fake_epoch = 0
    model.train()
    while step<=cfg['iter']:
        try:
            samples = next(train_loader_iter)
        except StopIteration:
            fake_epoch += 1
            if getattr(args, "distributed", False):
                train_sampler.set_epoch(fake_epoch)
            train_loader_iter = iter(train_loader)
            samples = next(train_loader_iter)

        optimizer.param_groups[0]['lr'] = 2*cfg['optimizer']['lr'] * (1 - float(step) / cfg['iter']) ** 0.9
        optimizer.param_groups[1]['lr'] = cfg['optimizer']['lr'] * (1 - float(step) / cfg['iter']) ** 0.9
        optimizer.zero_grad()
        input_image, label = samples['trans_image'], samples['label']
        input_image, label = input_image.cuda(non_blocking=True), label.cuda(non_blocking=True)
        patch_cls, patch_text, score, shadow_logit = model(input_image)

        bs = score.shape[0]  # 16
        shadow_ratio = label.float().mean(dim=[1, 2, 3]).clamp(0.0, 1.0)   # [B]
        pred_ratio = torch.sigmoid(score.squeeze(1))             # [B]
        w = (shadow_ratio + 0.05).pow(0.25)                       # 你也可以试 pow(1.0) 更激进

        loss1 = mywbce(patch_cls, label)
        loss2 = mywbce(patch_text, label)
        loss3 = mywbce(shadow_logit, label)
        loss4 = (w * F.smooth_l1_loss(pred_ratio, shadow_ratio, reduction='none')).mean()

        hold = 2000
        decay = 12000  # 你想用多少步降完自己设，比如 8000/10000
        if step < hold:
            lambda_cls, lambda_text = 1.0, 1.0
        else:
            t = min(1.0, (step - hold) / decay)
            lambda_cls  = 1.0 * (1 - t) + 0.05 * t
            lambda_text = 1.0 * (1 - t) + 0.05 * t

        loss = loss3 + lambda_cls * loss1 + lambda_text * loss2 + 0.25 * loss4


        predicts1 = torch.sigmoid(patch_cls)
        predicts2 = torch.sigmoid(patch_text)
        predicts3 = torch.sigmoid(shadow_logit)

        if is_main_process(args) and step % 20 == 0:
            now = time.perf_counter()
            elapsed = now - t0
            elapsed_min, elapsed_sec = int(elapsed // 60), int(elapsed % 60)
            logger.info(
                "step: %d loss: %.4f (final %.4f, cls %.4f*%.2f, text %.4f*%.2f, reg %.4f*0.25) | elapsed: %d min %d s"
                % (step, loss.item(),
                loss3.item(), loss1.item(), lambda_cls, loss2.item(), lambda_text, loss4.item(),
                elapsed_min, elapsed_sec)
            )
        if is_main_process(args) and step % 200 == 0:
            logger.info(f"lambdas: cls={lambda_cls:.4f}, text={lambda_text:.4f}")


        loss.backward()
        optimizer.step()

        if is_main_process(args):
            if step%50==0:
                save_mask(cfg['ckpt_image'], step, image_size, [samples['np_image'], samples['label'], (predicts1, predicts2, predicts3)])
                logger.info("save immediate mask: %d" % (step))
            model_to_save = model.module if isinstance(model, DDP) else model
            if step%500==0:
                save_ckpts(cfg['ckpt_model'], model_to_save, step, optimizer, "latest.pth")
            if step%2000==0:
                save_ckpts(cfg['ckpt_model'], model_to_save, step, optimizer, str(step)+'.pth')

        dist.barrier()   # 保存完再一起继续
        step+=1


    if is_main_process(args):
        total = time.perf_counter() - t0
        logger.info("Training finished. Total elapsed time: %.1fs (%.2f min)" % (total, total/60.0))
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="manual seed (default: random int)")
    parser.add_argument("--dataset", default="sbu", type=str, choices=["sbu", "istd"], help="choose dataset: sbu or istd")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(10000000)

    # 给 init_distributed 用的占位
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.distributed = False

    init_seeds(args.seed)
    init_distributed(args)

    try:
        main(args.seed, args)
    finally:
        cleanup_distributed()
