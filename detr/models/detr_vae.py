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
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim, pcl_backbone, depth_backbones):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            pointnet_dim: dimension of point cloud features from PointNet
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
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            if pcl_backbone is not None:
                self.input_proj_pointnet = nn.Linear(pcl_backbone.output_dim, hidden_dim)
            if depth_backbones is not None:
                self.input_proj_depth = nn.Conv2d(depth_backbones[0].num_channels, hidden_dim, kernel_size=1)
                self.depth_backbones = nn.ModuleList(depth_backbones)
                self.input_proj_depth = nn.Conv2d(depth_backbones[0].num_channels, hidden_dim, kernel_size=1)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        # 潜在向量z的维度
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        # 注册一个缓冲区pos_table，用于存储正弦位置编码表  缓存区是一种特殊的张量，它存储存的是一个固定的值，不会随着训练过程而改变。
        # get_sinusoid_encoding_table函数（未在代码中定义）可能用于生成正弦位置编码，这是Transformer架构中常用的一种技术，为序列中的每个位置提供唯一的表示
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else: 
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        if pcl_backbone is not None:
            self.pcl_backbone = pcl_backbone
            self.additional_pos_embed = nn.Embedding(3, hidden_dim) # learned position embedding for proprio, pointnet and latent
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        if depth_backbones is not None:
            # 为深度特征添加额外的位置编码
            self.depth_pos_embed = nn.Embedding(1, hidden_dim)

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

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None, pointcloud=None, depth_img=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        pointcloud: dict with keys:
            - xyz: (batch, num_points, 3) 点云坐标
            - rgb: (batch, num_points, 3) 点云颜色
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # cvae decoder
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0]  # 取最后一层特征
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            # 处理深度图特征
            if depth_img is not None and self.depth_backbones is not None:
                all_depth_features = []
                all_depth_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    depth_features, depth_pos = self.depth_backbones[cam_id](depth_img[:, cam_id])
                    depth_features = depth_features[0]  # 取最后一层特征
                    depth_pos = depth_pos[0]
                    # 投影到相同的隐藏维度
                    depth_features = self.input_proj_depth(depth_features)
                    all_depth_features.append(depth_features)
                    all_depth_pos.append(depth_pos)

                # 合并深度特征
                depth_src = torch.cat(all_depth_features, axis=3)
                depth_pos = torch.cat(all_depth_pos, axis=3)

            # 处理点云特征
            if pointcloud is not None and self.pcl_backbone is not None:
                pcl_features = self.pcl_backbone(pointcloud)
                pcl_input = self.input_proj_pointnet(pcl_features)
                pcl_input = pcl_input.unsqueeze(0)
            else:
                pcl_input = None

            # 处理机器人状态
            proprio_input = self.input_proj_robot_state(qpos)

            # 合并RGB特征
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # 根据是否有深度信息选择不同的transformer输入方式
            if depth_img is not None and self.depth_backbones is not None:
                # 调用修改后的transformer
                hs = self.transformer(
                    src=src,
                    mask=None,
                    query_embed=self.query_embed.weight,
                    pos_embed=pos,
                    latent_input=latent_input,
                    proprio_input=proprio_input,
                    additional_pos_embed=self.additional_pos_embed.weight,
                    pcl_input=pcl_input,
                    depth_src=depth_src,
                    depth_pos_embed=depth_pos
                )[0]
            else:
                # 不使用深度信息的原始调用方式
                hs = self.transformer(
                    src=src,
                    mask=None,
                    query_embed=self.query_embed.weight,
                    pos_embed=pos,
                    latent_input=latent_input,
                    proprio_input=proprio_input,
                    additional_pos_embed=self.additional_pos_embed.weight,
                    pcl_input=pcl_input
                )[0]
        else:
            # 处理只有状态输入的情况
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


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

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    depth_backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args, input_channels=3)  # RGB图像使用3通道
        backbones.append(backbone)
    
    if args.use_depth is False:
        depth_backbones = None
    else:
        for _ in args.depth_camera_names:
            depth_backbone = build_backbone(args, input_channels=1)  # 深度图使用1通道
            depth_backbones.append(depth_backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args)
        encoder = build_encoder(args)
        
    # 构建 PointNet 模型
    if args.use_pcd is False:
        pcl_backbone = None
    else:
        pcl_backbone = PointNet(
            n_coordinates=3,  # 点云坐标维度
            n_color=3,        # 点云颜色维度（如果有的话）
            output_dim=512,   # 输出特征维度
            hidden_dim=512,   # 隐藏层维度
            hidden_depth=3    # 网络层数
        )
        
    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
        depth_backbones=depth_backbones,
        pcl_backbone=pcl_backbone
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 7 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

