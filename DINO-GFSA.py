# main_unified_merged.py
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import os
import math
import warnings
import cv2
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

from transformers import AutoModel
from torchvision.transforms import v2
from pytorch_metric_learning import losses
from torch.cuda.amp import GradScaler, autocast

# ====================== 可选库导入 ======================
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
except ImportError:
    get_peft_model, LoraConfig, TaskType, PeftModel = None, None, None, None
    print("\n[WARNING] peft is not installed. LoRA components will fail.")

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("\n[WARNING] mamba-ssm is not installed. Mamba components will fail.")

warnings.filterwarnings("ignore")


# ===================================================================
# 1. 参数解析
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Unified DINOv3/ConvNeXt Training Script (No Validation)')
    
    # --- 核心选择 ---
    parser.add_argument('--backbone_type', default='dinov3', type=str, choices=['dinov3', 'convnext'],
                        help="Choose backbone architecture: 'dinov3' (ViT-based) or 'convnext' (Hybrid).")
    
    # --- 模式与路径 ---
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'], help='run mode')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0 or 0,1')
    parser.add_argument('--model_dir', default='/root/autodl-tmp/dinov3lvd', type=str,
                        help='Path to pre-trained backbone model')
    parser.add_argument('--train_dir', default='/root/autodl-tmp/DenseUAV/DenseUAV/train', type=str,
                        help='training data path')
    parser.add_argument('--test_dir', default='/root/autodl-tmp/DenseUAV/DenseUAV/test', type=str,
                        help='test data path')
    parser.add_argument('--save_dir', default='./models_merged_output', type=str,
                        help='directory to save models and visualizations')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume')

    # --- 模型超参数 ---
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    parser.add_argument('--dropout_p', default=0.6, type=float, help='dropout probability')
    parser.add_argument('--embedding_dim', default=512, type=int, help='output embedding dimension')
    parser.add_argument('--num_mamba_layers', default=2, type=int, help='Mamba layers in head')

    # --- 融合参数 (仅 DINOv3 使用) ---
    parser.add_argument('--fusion_layers', type=str, default='16,20,24',
                        help="[DINOv3 Only] Layers to fuse (low, mid, high).")

    # --- LoRA 参数 ---
    parser.add_argument('--lora_r', default=16, type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', default=32, type=int, help='LoRA alpha')

    # --- 训练超参数 ---
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=80, type=int, help='total epochs')
    parser.add_argument('--infonce_temp', default=0.07, type=float, help='InfoNCE temperature')
    parser.add_argument('--grad_clip_norm', default=1.0, type=float, help='Gradient clipping norm')

    # --- 学习率 ---
    parser.add_argument('--lr_backbone', default=2.8e-4, type=float, help="LR for Backbone/LoRA/Adapters")
    parser.add_argument('--lr_head', default=5e-5, type=float, help="LR for Mamba Head")

    # --- 测试模式 ---
    parser.add_argument('--test_mode', default=1, type=int, help='1: drone->sat, 2: sat->drone')

    return parser.parse_args()


# ===================================================================
# 2. 数据处理
# ===================================================================
def collate_fn_skip_corrupted(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

class PairedDenseUAVDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.drone_path = os.path.join(data_path, 'drone')
        self.satellite_path = os.path.join(data_path, 'satellite')
        self.image_pairs, self.num_classes = self._make_dataset()

    def _make_dataset(self):
        pairs = []
        class_ids = set()
        all_drone_images = glob(os.path.join(self.drone_path, '**', '*.JPG'), recursive=True) + \
                           glob(os.path.join(self.drone_path, '**', '*.jpg'), recursive=True)
        if not os.path.isdir(self.drone_path): return [], 0
        sorted_class_dirs = sorted(os.listdir(self.drone_path), key=int)
        class_to_idx = {class_id: i for i, class_id in enumerate(sorted_class_dirs)}
        for drone_img_path in all_drone_images:
            filename_stem = os.path.splitext(os.path.basename(drone_img_path))[0]
            class_id_str = os.path.basename(os.path.dirname(drone_img_path))
            satellite_img_path = os.path.join(self.satellite_path, class_id_str, f"{filename_stem}.tif")
            if os.path.exists(satellite_img_path):
                try:
                    label = class_to_idx[class_id_str]
                    pairs.append((drone_img_path, satellite_img_path, label))
                    class_ids.add(label)
                except: continue
        return pairs, len(class_ids)

    def __len__(self): return len(self.image_pairs)

    def __getitem__(self, idx):
        drone_img_path, sat_img_path, label = self.image_pairs[idx]
        try:
            drone_img = Image.open(drone_img_path).convert('RGB')
            sat_img = Image.open(sat_img_path).convert('RGB')
            if self.transform:
                drone_img = self.transform(drone_img)
                sat_img = self.transform(sat_img)
            return drone_img, sat_img, label
        except: return None

class SafeImageFolder(Dataset):
    def __init__(self, root, transform=None):
        internal_dataset = datasets.ImageFolder(root)
        self.samples = internal_dataset.samples
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = default_loader(path)
            if self.transform is not None: sample = self.transform(sample)
            return sample, target
        except: return None
    def __len__(self): return len(self.samples)


# ===================================================================
# 3. 核心组件 (统一命名)
# ===================================================================

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        x = x.permute(0, 2, 1).clamp(min=self.eps)
        p = torch.clamp(self.p, min=1.0)
        x = F.avg_pool1d(x.pow(p), kernel_size=x.size(2)).pow(1. / p)
        return x.squeeze(2)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class StackedMambaHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_mamba_layers=3, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, input_dim)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_mamba_layers)
        ])
        self.pooling = GeMPooling()
        self.proj_out = nn.Linear(input_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.mamba_blocks: x = block(x)
        pooled = self.pooling(x)
        return self.norm_out(self.proj_out(pooled))

# --- 核心融合模块 (原 AdvancedHierarchicalFusion/SemanticGatedResidualFusion) ---
class SemanticGatedResidualFusion(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 中层提炼
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True)
        )
        # 语义提取 (SE Block)
        self.high_level_squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True)
        )
        # 细节门控 (Spatial Attention)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False), nn.Sigmoid()
        )

    def forward(self, low_level_map, mid_level_map, high_level_map):
        B, C, _, _ = high_level_map.shape
        # 1. 主干融合 (Mid + High)
        refined_mid = mid_level_map + self.refine_conv(mid_level_map)
        semantic_sig = self.high_level_squeeze(high_level_map).view(B, C)
        channel_w = self.excitation(semantic_sig).view(B, C, 1, 1)
        calibrated_mid = refined_mid * channel_w
        fused_mid_high = self.fusion_conv(calibrated_mid)
        
        # 2. 细节注入 (Low guided by Fused)
        gate_map = self.detail_gate(fused_mid_high)
        gated_low = low_level_map * gate_map
        
        return fused_mid_high + gated_low

# --- 核心聚合模块 (统一整合了 UnifiedHead 和 SpatialMambaJointAggregation) ---
class SequentialAggregation(nn.Module):
    def __init__(self, backbone_dim, output_dim, num_mamba_layers):
        super().__init__()
        # 这里接收融合后的特征图，转换为序列输入 Mamba
        self.mamba_head = StackedMambaHead(backbone_dim, output_dim, num_mamba_layers=num_mamba_layers)

    def forward(self, fused_map):
        # fused_map: [B, C, H, W] -> [B, H, W, C]
        x_permuted = fused_map.permute(0, 2, 3, 1)
        B, H, W, C = x_permuted.shape
        # Flatten spatial dimensions: [B, H*W, C]
        x_seq = x_permuted.reshape(B, H * W, C)
        return self.mamba_head(x_seq)

class FinalProjectorHead(nn.Module):
    def __init__(self, backbone_dim, embedding_dim, dropout_p, num_mamba_layers):
        super().__init__()
        self.aggregator = SequentialAggregation(backbone_dim, embedding_dim, num_mamba_layers)

    def forward(self, fused_map):
        embed = self.aggregator(fused_map)
        return (embed, None, None), (None, None)


# ===================================================================
# 4. 统一模型主类
# ===================================================================
class UnifiedCrossViewModel(nn.Module):
    def __init__(self, backbone_type, model_dir, embedding_dim, dropout_p, num_mamba_layers, 
                 fusion_layers, h, w, lora_r, lora_alpha):
        super().__init__()
        self.backbone_type = backbone_type
        self.h = h
        self.w = w
        self.fusion_layers = fusion_layers # List[int] for ViT
        
        print(f"Loading backbone type: {backbone_type} from {model_dir}")
        base_backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        
        # --- Backbone 特定初始化 ---
        if backbone_type == 'dinov3':
            # ViT 逻辑
            self.backbone_dim = base_backbone.config.hidden_size
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"]
            )
            self.backbone = get_peft_model(base_backbone, peft_config)
            
        elif backbone_type == 'convnext':
            # ConvNeXt 逻辑
            # ConvNeXt 的 LoRA 目标层列表 (非常长，简化展示，实际运行时需完整)
            targets = []
            # 自动生成 denseconv.py 中的长列表
            for stage in range(4): # stages 0-3
                for layer in range(30): # 假设最大层数覆盖
                    targets.append(f'stages.{stage}.layers.{layer}.pointwise_conv1')
                    targets.append(f'stages.{stage}.layers.{layer}.pointwise_conv2')
            
            peft_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, target_modules=targets,
                lora_dropout=0.1, bias="none"
            )
            self.backbone = get_peft_model(base_backbone, peft_config)
            
            # ConvNeXt 特有的 Adapters (用于对齐通道)
            # 假设 ConvNeXt 不同阶段输出通道为 384, 768, 1536 (Large/Base variant)
            # 需要根据具体模型调整，这里沿用 denseconv.py 的设定
            c2_ch, c3_ch, c4_ch = 384, 768, 1536 
            self.target_channel = 768
            self.backbone_dim = self.target_channel
            
            self.adapter_c2 = nn.Conv2d(c2_ch, self.target_channel, kernel_size=1, bias=False)
            self.adapter_c3 = nn.Conv2d(c3_ch, self.target_channel, kernel_size=1, bias=False)
            self.adapter_c4 = nn.Conv2d(c4_ch, self.target_channel, kernel_size=1, bias=False)
        
        # --- 通用部分 ---
        self.fusion_module = SemanticGatedResidualFusion(channel=self.backbone_dim)
        
        self.projector = FinalProjectorHead(
            backbone_dim=self.backbone_dim,
            embedding_dim=embedding_dim,
            dropout_p=dropout_p,
            num_mamba_layers=num_mamba_layers
        )
        
        self.backbone.print_trainable_parameters()

    def _extract_features(self, images):
        if self.backbone_type == 'dinov3':
            # DINOv3 (ViT) 提取逻辑
            outputs = self.backbone(pixel_values=images, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            B, _, C = hidden_states[0].shape
            # 计算 Patch 数量
            num_special = hidden_states[0].shape[1] - (self.h // 16) * (self.w // 16)
            H = W = int((hidden_states[0].shape[1] - num_special) ** 0.5)
            
            l_idx, m_idx, h_idx = self.fusion_layers
            
            low = hidden_states[l_idx][:, num_special:, :].transpose(1, 2).reshape(B, C, H, W)
            mid = hidden_states[m_idx][:, num_special:, :].transpose(1, 2).reshape(B, C, H, W)
            high = hidden_states[h_idx][:, num_special:, :].transpose(1, 2).reshape(B, C, H, W)
            return low, mid, high

        elif self.backbone_type == 'convnext':
            # ConvNeXt 提取逻辑
            # 注意: AutoModel ConvNeXt 可能需要用 inputs 或者是 pixel_values
            # 这里为了兼容，直接调用 self.backbone(images) 并假设 images 是 tensor
            outputs = self.backbone(images, output_hidden_states=True)
            states = outputs.hidden_states
            
            # 假设 states[2], [3], [4] 对应所需的层级
            raw_low = states[2]
            raw_mid = states[3]
            raw_high = states[4]
            
            low = self.adapter_c2(raw_low)
            mid_adapted = self.adapter_c3(raw_mid)
            high_adapted = self.adapter_c4(raw_high)
            
            # 插值对齐尺寸
            target_size = low.shape[2:]
            mid = F.interpolate(mid_adapted, size=target_size, mode='bilinear', align_corners=False)
            high = F.interpolate(high_adapted, size=target_size, mode='bilinear', align_corners=False)
            
            return low, mid, high

    def forward(self, drone_images=None, satellite_images=None):
        images = drone_images if drone_images is not None else satellite_images
        if images is None: return None, None, None

        low, mid, high = self._extract_features(images)
        fused_map = self.fusion_module(low, mid, high)
        final_out, aux_out = self.projector(fused_map)
        
        return final_out, aux_out, fused_map


# ===================================================================
# 5. 辅助函数
# ===================================================================
def plot_loss_curve(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(img.device)
    return img.index_select(3, inv_idx)

def extract_feature(model, dataloader, view, device, embedding_dim):
    features = torch.FloatTensor()
    model.eval()
    for data in tqdm(dataloader, desc=f"Extracting {view}"):
        img, _ = data
        if img.size(0) == 0: continue
        ff = torch.zeros(img.size(0), embedding_dim).to(device)
        for i in range(2): # TTA: Original + Flip
            inp = fliplr(img.to(device)) if i == 1 else img.to(device)
            with torch.no_grad(), autocast():
                if view == 'drone': (emb,_,_),_,_ = model(drone_images=inp)
                else: (emb,_,_),_,_ = model(satellite_images=inp)
            ff += emb
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.cpu()), 0)
    return features

def get_id(samples):
    labels, paths = [], []
    for path, _ in samples:
        labels.append(int(os.path.basename(os.path.dirname(path))))
        paths.append(path)
    return labels, paths


# ===================================================================
# 6. 主逻辑
# ===================================================================
if __name__ == "__main__":
    opt = parse_args()
    if Mamba is None or get_peft_model is None:
        raise SystemExit("Error: mamba-ssm and peft are required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Backbone: {opt.backbone_type}")

    # 数据变换
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = v2.Compose([
        v2.ToImage(), v2.Resize((opt.h, opt.w), antialias=True), v2.TrivialAugmentWide(),
        v2.RandomHorizontalFlip(p=0.5), v2.ColorJitter(0.3, 0.3),
        v2.RandomErasing(p=0.2), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])
    test_transform = v2.Compose([
        v2.ToImage(), v2.Resize((opt.h, opt.w), antialias=True),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)
    ])

    # 解析融合层索引 (仅 DINOv3)
    try:
        fusion_indices = [int(x) for x in opt.fusion_layers.split(',')]
    except: fusion_indices = [16, 20, 24]

    # 初始化模型
    model = UnifiedCrossViewModel(
        backbone_type=opt.backbone_type,
        model_dir=opt.model_dir,
        embedding_dim=opt.embedding_dim,
        dropout_p=opt.dropout_p,
        num_mamba_layers=opt.num_mamba_layers,
        fusion_layers=fusion_indices,
        h=opt.h, w=opt.w,
        lora_r=opt.lora_r, lora_alpha=opt.lora_alpha
    )
    model.to(device)
    
    # 编译优化
    if hasattr(torch, 'compile'):
        try: model = torch.compile(model)
        except: pass

    # ================= TRAIN MODE =================
    if opt.mode == 'train':
        print("\n--- Training Mode (No Validation) ---")
        os.makedirs(opt.save_dir, exist_ok=True)
        
        train_ds = PairedDenseUAVDataset(opt.train_dir, transform=train_transform)
        train_loader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, 
                                  num_workers=8, drop_last=True, collate_fn=collate_fn_skip_corrupted)
        
        loss_func = losses.NTXentLoss(temperature=opt.infonce_temp).to(device)
        scaler = GradScaler()
        
        # 参数分组与优化器
        access_model = model.module if isinstance(model, nn.DataParallel) else model
        params_backbone = [p for p in access_model.backbone.parameters() if p.requires_grad]
        
        # 如果是 ConvNeXt，还需要加入 adapters 的参数
        if opt.backbone_type == 'convnext':
            params_backbone += list(access_model.adapter_c2.parameters())
            params_backbone += list(access_model.adapter_c3.parameters())
            params_backbone += list(access_model.adapter_c4.parameters())
            
        params_head = list(access_model.projector.parameters()) + \
                      list(access_model.fusion_module.parameters())
        
        optimizer = optim.AdamW([
            {'params': params_backbone, 'lr': opt.lr_backbone},
            {'params': params_head, 'lr': opt.lr_head}
        ], weight_decay=5e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
        history = {'train_loss': []}
        
        # Resume 逻辑
        start_epoch = 0
        if opt.resume and os.path.exists(opt.resume):
            ckpt = torch.load(opt.resume, map_location=device)
            access_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, opt.epochs):
            model.train()
            ep_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs}")
            
            for data in pbar:
                d_img, s_img, labels = [d.to(device) for d in data]
                if d_img.size(0) < 2: continue
                
                optimizer.zero_grad()
                with autocast():
                    (d_emb,_,_),_,_ = model(drone_images=d_img)
                    (s_emb,_,_),_,_ = model(satellite_images=s_img)
                    loss = loss_func(torch.cat([d_emb, s_emb]), torch.cat([labels, labels]))
                
                if torch.isnan(loss): continue
                scaler.scale(loss).backward()
                if opt.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                
                ep_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            scheduler.step()
            avg_loss = ep_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
            
            # 保存 Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': access_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, os.path.join(opt.save_dir, 'checkpoint_latest.pth'))
            
            plot_loss_curve(history, os.path.join(opt.save_dir, 'loss_curve.png'))

        # 保存最终模型
        torch.save(access_model.state_dict(), os.path.join(opt.save_dir, 'model_final.pth'))
        print("Training Finished.")

    # ================= TEST MODE =================
    elif opt.mode == 'test':
        print("\n--- Test Mode ---")
        # 加载权重
        path = os.path.join(opt.save_dir, 'model_final.pth')
        if not os.path.exists(path):
            path = os.path.join(opt.save_dir, 'checkpoint_latest.pth')
            print(f"model_final.pth not found, trying {path}")
            ckpt = torch.load(path, map_location=device)
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = torch.load(path, map_location=device)
        
        # 处理可能的 DataParallel 包装
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        # 宽容加载 (因为 backbone 差异)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. {msg}")

        q_name, g_name = ('drone', 'satellite') if opt.test_mode == 1 else ('satellite', 'drone')
        dsets = {x: SafeImageFolder(os.path.join(opt.test_dir, x), test_transform) for x in [g_name, q_name]}
        loaders = {x: DataLoader(dsets[x], batch_size=opt.batchsize, shuffle=False, num_workers=8, 
                                 collate_fn=collate_fn_skip_corrupted) for x in [g_name, q_name]}

        q_feat = extract_feature(model, loaders[q_name], q_name, device, opt.embedding_dim)
        g_feat = extract_feature(model, loaders[g_name], g_name, device, opt.embedding_dim)

        g_lbl, g_path = get_id(dsets[g_name].samples)
        q_lbl, q_path = get_id(dsets[q_name].samples)

        result = {'gallery_f': g_feat.numpy(), 'gallery_label': g_lbl, 'gallery_path': g_path,
                  'query_f': q_feat.numpy(), 'query_label': q_lbl, 'query_path': q_path}
        
        out_name = f"result_{opt.backbone_type}_mode{opt.test_mode}.mat"
        scipy.io.savemat(out_name, result)
        print(f"Saved results to {out_name}")