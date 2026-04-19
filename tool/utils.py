import os
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
import skimage.io as io
from pathlib import Path
import torch.nn.functional as F
import torch.distributed as dist

import torch
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  [B,1,H,W]  raw logits
        targets: [B,1,H,W]  0/1 mask
        """
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction="mean"):
    """
    多分类 Focal Loss
    logits: [B, C]  => 你的 score
    targets: [B]    => area_cls (取值范围 0-3)
    alpha:   类别平衡参数
    gamma:   难例聚焦参数
    """
    ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [B]
    pt = torch.exp(-ce_loss)  # pt = softmax prob of correct class

    focal = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    return focal


def exist_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)



def init_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print("os.environ[\"WORLD_SIZE\"]: ", os.environ["WORLD_SIZE"])
        print("os.environ[\"RANK\"]: ", os.environ["RANK"])
        print("os.environ[\"LOCAL_RANK\"]: ", os.environ["LOCAL_RANK"])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    dist_url = 'env://'
    args.dist_backend = 'nccl'  # 通信后端
    print('| distributed init (rank {}): {}'.format(args.rank, dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()
    args.distributed = True


def is_main_process(args):
    return (not getattr(args, "distributed", False)) or args.rank == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

'''  initial seed  '''
def init_seeds(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False


class MyBceloss12_n(nn.Module):
    def __init__(self):
        super(MyBceloss12_n, self).__init__()

    def forward(self, pred, gt, dst1, dst2):
        eposion = 1e-10
        sigmoid_dst1 = torch.sigmoid(dst1)
        sigmoid_dst2 = torch.sigmoid(dst2)
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos/(count_pos+count_neg)
        dst_loss = beta*(1+dst2)*gt*F.binary_cross_entropy_with_logits(pred, gt, reduction='none') + \
                   (1+dst1)*(1-gt)*F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        bce2_lss = torch.mean(dst_loss)
        loss = beta_back*bce1(pred, gt) + beta_back*bce2_lss
        return loss


class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()

    def forward(self, pred, gt):
        eposion = 1e-10
        # sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)
        return loss


def save_mask(path_dir, step, image_size, img_lists: tuple):
    data, label, predict = img_lists

    # process data and label
    data, label= (data.numpy()*255).astype('uint8'), (label.numpy()*255).astype('uint8')
    label = np.tile(label, (3,1,1))
    h, w = image_size, image_size

    # process predicts
    predicts = []
    for pred in predict:
        pred = (np.tile(pred.cpu().data * 255,(3,1,1))).astype('uint8')
        predicts.append(pred)

    # save image
    gen_num = (2, 1)  # save two example images
    per_img_num = len(predicts)+2
    img = np.zeros((gen_num[0]*h, gen_num[1]*(per_img_num)*w, 3)).astype('uint8')
    for _ in img_lists:
        for i in range(gen_num[0]):  # i row
            row = i * h
            for j in range(gen_num[1]):  # j col
                idx = i * gen_num[1] + j
                # save data && gt mask
                pred_mask_list = [p[idx] for p in predicts]
                tmp_list = [data[idx], label[idx]] + pred_mask_list
                for k in range(per_img_num):
                    col = (j * 3 + k) * w
                    tmp = np.transpose(tmp_list[k], (1, 2, 0))
                    # print(tmp.shape)
                    img[row: row+h, col: col+w] = tmp

    img_file = os.path.join(path_dir, '%d.jpg'%(step))
    io.imsave(img_file, img)




def load_checkpoints(model, model_dir, name):
    ckp_path = os.path.join(model_dir, name)
    try:
        # obj = torch.load(ckp_path, map_location='cpu')
        obj = torch.load(ckp_path)
    except FileNotFoundError:
        return print("File Not Found")
    model.load_state_dict(obj['model'])


def save_ckpts(path, model, step, optimizer, name):
    ckpt_path = os.path.join(path, name)
    obj = {
        'model': model.state_dict(),
        'clock': step,
        'optim': optimizer.state_dict(),
    }
    torch.save(obj, ckpt_path)


def load_config(file_name):
    path = Path(__file__).parent.parent/"cfg"/file_name
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    import os.path as osp
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)