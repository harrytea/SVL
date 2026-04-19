import os
import torch
from torch import nn
from torch.nn import functional as F
from CLIP.tokenizer import tokenize as tokenize
from .adaptor import VisionContextAdapter, TextAdapter


class ShadowDINO(nn.Module):
    def __init__(self, clip_model, dino, device):
        super().__init__()
        self.device = device

        self.clip_model = clip_model
        self.dinov3 = dino
        self.cls_adapter = nn.ModuleList([VisionContextAdapter(c_in = 1024) for _ in range(4)])
        self.patch_adapter = nn.ModuleList([VisionContextAdapter(c_in = 1024) for _ in range(4)])
        self.text_adapter = nn.ModuleList([TextAdapter(c_in = 768) for _ in range(1)])

        # ------------------------------------------------------
        self.log_scale_patch_text = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        self.log_scale_patch_cls  = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        self.log_scale_global     = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        # ------------------------------------------------------

        with torch.no_grad():
            self.cached_text_features = self.encode_text_with_prompt_ensemble(self.clip_model, "scene", self.device, '').unsqueeze(0)
        self.layer_weights = nn.Parameter(torch.zeros(4))  # 4 层 DINO 的可学习融合权重



        self.skip17_proj = nn.Conv2d(1024, 16, kernel_size=1, bias=False)  # 用第 17 层 patch token 做 skip：1024 -> 16 通道（你也可改 8/32）
        self.skip17_gate = nn.Sequential(nn.Conv2d(16, 16, 1), nn.Sigmoid())

        self.refine_head = nn.Sequential(
            # ===== 32x32 =====
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # ===== 32 -> 64 =====
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # ===== 64 -> 128 =====
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # ===== 128 -> 256 =====
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            # ===== 256 -> 512 =====
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),

            # ===== output =====
            nn.Conv2d(8, 1, kernel_size=1)
        )

        self.global_head = nn.Linear(1, 1)  # 输入为 shadow 相似度标量，输出 4 类面积桶


    def forward(self, image, ):
        bs, c, h, w = image.shape
        text_features = self.get_adjusted_text_features().expand(bs, -1, -1)  # [B,768,1]
        cls_token, patch_tokens, low_features = self.get_feature_dinov3(bs, image, self.device, self.dinov3)

        scale_patch_text = torch.exp(self.log_scale_patch_text)
        scale_patch_cls  = torch.exp(self.log_scale_patch_cls)
        scale_global     = torch.exp(self.log_scale_global)

        shadow_patch_cls_32_all, shadow_patch_text_32_all, shadow_scores_all = [], [], []
        for i in range(4):
            cls_features = F.normalize(self.cls_adapter[i](cls_token[i]), p=2, dim=-1, eps=1e-6)
            patch_features = F.normalize(self.patch_adapter[i](patch_tokens[i]), p=2, dim=-1, eps=1e-6)

            B, N, _ = patch_features.shape
            H = int(N ** 0.5)

            # cls_features: [B,1,C], patch_features: [B,N,C]
            shadow_patch_cls_32 = scale_patch_cls * (patch_features * cls_features).sum(dim=-1, keepdim=True)  # [B,N,1]
            shadow_patch_cls_32 = shadow_patch_cls_32.transpose(1,2).view(B,1,H,H)

            # patch * text → [B,2,32,32]
            shadow_patch_text_32 = scale_patch_text * (patch_features @ text_features)  # [B,N,1]
            shadow_patch_text_32 = shadow_patch_text_32.permute(0,2,1).view(B,1,H,H)


            # cls * text → [B,1,2]
            shadow_score = scale_global * (cls_features @ text_features)

            shadow_patch_cls_32_all.append(shadow_patch_cls_32)
            shadow_patch_text_32_all.append(shadow_patch_text_32)
            shadow_scores_all.append(shadow_score)

        # 多层 32×32 map 的可学习加权融合
        weights = torch.softmax(self.layer_weights, dim=0)
        shadow_cls_32  = sum(w * m for w, m in zip(weights, shadow_patch_cls_32_all))
        shadow_text_32 = sum(w * m for w, m in zip(weights, shadow_patch_text_32_all))
        score          = sum(w * s for w, s in zip(weights, shadow_scores_all))
        score = score.squeeze(-1)   # [B,1]
        score_shadow = self.global_head(score)



        # ========== UNet-lite skip: 用第 17 层 patch token ==========
        # 你的 layers=(5,11,17,23)，因此 17 对应 patch_tokens[2]
        skip_tokens = low_features              # [B, 1024, 1024]  (N=1024)
        B, N, C = skip_tokens.shape                # C=1024
        H = int(N ** 0.5)                          # 32

        skip_map = skip_tokens.transpose(1, 2).contiguous().view(B, C, H, H)  # [B,1024,32,32]
        skip_map = self.skip17_proj(skip_map)       # [B,16,32,32]
        skip_map = skip_map * self.skip17_gate(skip_map)
        # ===========================================================
        feat_32 = torch.cat([shadow_cls_32, shadow_text_32, skip_map], dim=1)  # [B, 2+16, 32, 32]
        shadow_logit = self.refine_head(feat_32)



        # 上采样到 512×512
        patch_cls    = F.interpolate(shadow_cls_32,    size=512, mode='bilinear', align_corners=False)  # 原先 cls prior
        patch_text   = F.interpolate(shadow_text_32,   size=512, mode='bilinear', align_corners=False)  # 原先 text map

        return patch_cls, patch_text, score_shadow, shadow_logit


    def get_adjusted_text_features(self):
        feat = self.cached_text_features[:, :, 0]   # [1,768]
        adj  = self.text_adapter[0](feat)           # [1,768]
        text_features = adj.unsqueeze(2)            # [1,768,1]
        return F.normalize(text_features, p=2, dim=1, eps=1e-6)


    def encode_text_with_prompt_ensemble(self, model, obj, device, cnn):
        prompt_shadow = ['shadowed {}', '{} with shadow', 'cast shadow on {}', 'harsh shadow on {}', 'soft shadow over {}']
        prompt_state = [prompt_shadow]
        prompt_templates = ['a photo of a {}.', 'a close-up photo of a {}.', 'a detailed photo of a {}.', 'an outdoor photo of a {}.', 'an indoor photo of a {}.', 'a {} under direct sunlight.', 'a {} in shade.', 'a {} under an overcast sky.', 'a {} at sunset.', 'a {} at noon.', 'a {} under side lighting.', 'a {} under top lighting.', 'a {} under backlighting.', 'a {} under a street lamp.', 'a {} near a window with light.', 'a bright scene with a {}.', 'a dim scene with a {}.', 'a high-contrast scene of a {}.', 'a low-contrast scene of a {}.', 'a natural light photo of a {}.', 'an artificial light photo of a {}.', 'a good photo of a {}.', 'a clear photo of a {}.', 'a bright photo of a {}.', 'a dark photo of a {}.', 'a realistic photo of a {}.', 'a photo of the {}.', 'a photo of one {}.', 'a small {} in the scene.', 'a large {} in the scene.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.']

        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
                    
            prompted_sentence = tokenize(prompted_sentence).to(device)
            class_embeddings = model.encode_text(text = prompted_sentence, cnn = cnn,)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).to(device)
        return text_features


    def get_feature_dinov3(self, bs, images, device, dino, layers=(5, 11, 17, 23), forward_bs=4, skip_idx=2):
        patch_tokens_dict = {i: [] for i in layers}
        cls_tokens_dict = {i: [] for i in layers}
        
        target_skip_features = []
        for start in range(0, bs, forward_bs):
            end = min(start + forward_bs, bs)
            image = images[start:end].to(device, non_blocking=True)
            patch_embedding, features = dino(image)

            if skip_idx == 'patch_embedding':
                # Patch Embedding 通常没有 CLS token 和 Register tokens，直接使用
                print("patch")
                tgt = patch_embedding
                tgt = tgt[:, 5:, :] 
                tgt = F.layer_norm(tgt, (tgt.shape[-1],))
                target_skip_features.append(tgt)
            elif isinstance(skip_idx, int) and skip_idx >= 0:
                toks = features[skip_idx]
                patches = toks[:, 5:, :] 
                patches = F.layer_norm(patches, (patches.shape[-1],))
                target_skip_features.append(patches)

            for i in layers:
                toks = features[i]
                patches = toks[:, 5:, :]
                patches = F.layer_norm(patches, (patches.shape[-1],))

                patch_tokens_dict[i].append(patches)
                cls_tokens_dict[i].append(toks[:, 0:1, :])

        patch_tokens = [torch.cat(patch_tokens_dict[i], dim=0) for i in layers]
        cls_tokens = [torch.cat(cls_tokens_dict[i], dim=0) for i in layers]
        
        if skip_idx == 'patch_embedding' or (isinstance(skip_idx, int) and skip_idx >= 0):
            low_features = torch.cat(target_skip_features, dim=0)
        else:
            low_features = patch_tokens[2]

        return cls_tokens, patch_tokens, low_features