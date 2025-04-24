# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .pointnet import PointNet

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim, pcl_backbone):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            encoder: encoder for VAE
            state_dim: robot state dimension of the environment
            num_queries: number of object queries
            vq: whether to use vector quantization
            vq_class: number of classes for vector quantization
            vq_dim: dimension for vector quantization
            action_dim: dimension of action space
            pcl_backbone: PointNet backbone for point cloud processing
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer.d_model #隐藏层维度
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 输入投影层
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        if pcl_backbone is not None:
            self.input_proj_pointnet = nn.Linear(pcl_backbone.output_dim, hidden_dim)
            self.pcl_pos_embed = nn.Embedding(1, hidden_dim)       # 点云的位置编码
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # 只需要2个位置编码：latent和proprio
        self.pcl_backbone = pcl_backbone

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)

        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        # 注册一个缓冲区pos_table，用于存储正弦位置编码表  缓存区是一种特殊的张量，它存储存的是一个固定的值，不会随着训练过程而改变。
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, pointcloud, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        pointcloud: dict with keys:
            - xyz: (batch, num_points, 3) 点云坐标
            - rgb: (batch, num_points, 3) 点云颜色
        actions: batch, seq, action_dim
        is_pad: batch, seq 用于mask padding的位置
        vq_sample: VQ-VAE采样结果
        """
        # 获取编码器输出
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # 处理点云特征
        if self.pcl_backbone is not None:
            # 使用PointNet提取点云特征
            pcl_features = self.pcl_backbone(pointcloud)
            src = self.input_proj_pointnet(pcl_features)  # [bs, dim]
            src = src.unsqueeze(0)  # [1, bs, dim]

            # 处理机器人状态
            proprio_input = self.input_proj_robot_state(qpos)  # [bs, dim]

            # 为各输入准备位置编码
            bs = src.shape[1]
            
            # 点云位置编码
            pcl_pos_embed = self.pcl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [1, bs, dim]
            
            # transformer处理
            hs = self.transformer(
                src=src,  # 点云特征作为src输入
                mask=None,
                query_embed=self.query_embed.weight,
                pos_embed=pcl_pos_embed,  # 点云的位置编码
                latent_input=latent_input,
                proprio_input=proprio_input,
                additional_pos_embed=self.additional_pos_embed.weight  # 用于latent和proprio的位置编码
            )[0]
        else:
            raise ValueError("PointNet backbone is required for point cloud processing")

        # 预测动作和padding mask
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 7 # TODO hardcode

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args)
        encoder = build_encoder(args)
        
    # 构建 PointNet 模型
    pcl_backbone = PointNet(
        n_coordinates=3,  # 点云坐标维度
        n_color=3,        # 点云颜色维度（如果有的话）
        output_dim=512,   # 输出特征维度
        hidden_dim=512,   # 隐藏层维度
        hidden_depth=3    # 网络层数
    )
        
    model = DETRVAE(
        transformer,
        encoder,
        state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
        pcl_backbone=pcl_backbone
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model