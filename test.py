import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import argparse
import numpy as np
from tool.utils import init_distributed, init_seeds, is_main_process, cleanup_distributed
from tool.utils import mkdir_or_exist, load_config, save_ckpts, save_mask, load_checkpoints
from tqdm import tqdm
from PIL import Image
from tool.sbudata import SBUTestDataset
from tool.istddata import ISTDTestDataset
from tool.ucfdata import UFCTestDataset
from tool.misc import crf_refine
from CLIP.clip import create_model
from model.shadowdino import ShadowDINO
from dinov3.backbones import dinov3_vitl16
from skimage import io


# SBU
sbu_image = r"/SSD/wangyh/shadow/shadowdata/SBU-shadow/SBU-shadow/SBU-Test/ShadowImages"
sbu_mask  = r"/SSD/wangyh/shadow/shadowdata/SBU-shadow/SBU-shadow/SBU-Test/ShadowMasks"
# ISTD
istd_image = r"/SSD/wangyh/shadow/shadowdata/ISTD/test/test_A"
istd_mask  = r"/SSD/wangyh/shadow/shadowdata/ISTD/test/test_B"
# UCF
ucf_image = r'/SSD/wangyh/shadow/shadowdata/UCF_shadow/InputImages'
ucf_mask = r'/SSD/wangyh/shadow/shadowdata/UCF_shadow/GroundTruth'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir',   default='./sbu/ckpt', help='directory for checkpoints')
parser.add_argument('--save_dir',   default='./results/',  help='directory for checkpoints')
parser.add_argument('--batch_size', default=1, type=int, help='number of samples in one batch')
parser.add_argument('--image_size',   default=512,  help='directory for checkpoints')
parser.add_argument('--ee_start', type=int, default=12000)
parser.add_argument('--ee_end',   type=int, default=20001)
parser.add_argument('--ee_step',  type=int, default=2000)
parser.add_argument('--dataset', type=str, default='sbu', choices=['sbu', 'istd', 'ucf'], )
args = parser.parse_args()



def main():
    if args.dataset == 'sbu':
        CurrentDataset = SBUTestDataset
        test_img_path = sbu_image
        config_file = "sbu.yaml"
    elif args.dataset == 'istd':
        CurrentDataset = ISTDTestDataset
        test_img_path = istd_image
        config_file = "istd.yaml" 
    elif args.dataset == 'ucf':
        CurrentDataset = UFCTestDataset
        test_img_path = ucf_image
        config_file = "sbu.yaml"
    print(f"Testing on {args.dataset.upper()} dataset...")

    for ee in range(args.ee_start, args.ee_end, args.ee_step):
        cfg = load_config(config_file)
        args.save_dirs = [
            f'./results/{args.dataset}_pc/{ee}',
            f'./results/{args.dataset}_pt/{ee}',
            f'./results/{args.dataset}_logit/{ee}'
        ]
        for save_dir in args.save_dirs:
            print(save_dir)
            os.makedirs(save_dir, exist_ok=True)

        test_dataset    = CurrentDataset(args.image_size)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=False)

        print("build model...")
        Dinov3_model_path = './dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        Dino_model = dinov3_vitl16()
        state_dict = torch.load(Dinov3_model_path, map_location='cpu', weights_only=True)
        rdict = Dino_model.load_state_dict(state_dict, strict=True)
        clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device="cuda", pretrained='openai', require_pretrained=True)
        model = ShadowDINO(clip_model, Dino_model, device="cuda")
        load_checkpoints(model, args.ckpt_dir, str(ee)+".pth")
        model.cuda().eval()

        for i, (batch, file_path) in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                O, B,= batch['trans_image'], batch['label']
                O, B = O.cuda(), B.cuda()
                shadow_patch_cls_mean, shadow_patch_text_mean, global_shadow_score, shadow_logit = model(O)

                # 定义一个帮助函数来避免重复代码 (可选优化，让代码更整洁)
                def save_result(prediction, save_index):
                    predict = torch.sigmoid(prediction)
                    image = Image.open(os.path.join(test_img_path, file_path[0])).convert('RGB')
                    final = Image.fromarray((predict.cpu().data * 255).numpy().astype('uint8')[0,0,:,:])
                    final = np.array(final.resize(image.size))
                    final_crf = crf_refine(np.array(image), final)
                    io.imsave(os.path.join(args.save_dirs[save_index], file_path[0]), final_crf)

                save_result(shadow_patch_cls_mean, 0)
                save_result(shadow_patch_text_mean, 1)
                save_result(shadow_logit, 2)

if __name__ == '__main__':
    main()