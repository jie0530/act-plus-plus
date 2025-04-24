action_dim=24
state_dim=24
latent_dim=32
action_head: in_features=512 out_features=24
cls_embed(1,512) embedding_dim=512 num_embedding=1
encoder_action_proj: in_features=24 out_features=512
encoder_joint_proj: in_features=24 out_features=512
input_proj:  in_channels=512  out_channels=512
input_proj_robot_state: in_features=24 out_features=512
latent_out_proj: in_features=32 out_features=512
latent_proj: in_features=512 out_features=64
quert_embed(10,512) embedding_dim=512 num_embedding=10
num_queries=10

action_data: shape(1x10x26)
image_data: shape(1x3x3x480x640)
qpose_data: shape(1x24)
is_pad: shape(1x10)

ACT_origin
action_data: shape(64x60x16) ---64:batch_size; 60:chunk_size; 16:action_dim
image_data: shape(64x3x3x480x640)
qpose_data: shape(64x14)
is_pad: shape(64x60)
action_dim=16
state_dim=14
latent_dim=32
action_head: in_features=512 out_features=16
cls_embed(1,512) embedding_dim=512 num_embedding=1
encoder_action_proj(16,512): in_features=16 out_features=512
encoder_joint_proj(14,512): in_features=14 out_features=512
input_proj:  in_channels=512  out_channels=512
input_proj_robot_state: in_features=14 out_features=512
latent_out_proj: in_features=32 out_features=512
latent_proj: in_features=512 out_features=64
quert_embed(60,512) embedding_dim=512 num_embedding=60
is_pad_head(512,1): in_features=512 out_features=1
num_queries=60
kl_weight=30
