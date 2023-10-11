#!/usr/bin/env python3

import torch
import torch.nn as nn
import timm
from transformers import T5Model, T5Tokenizer, BertModel, BertTokenizer
import clip
from utils.config_utils import *
import numpy as np
from prediction.fuse_helper import *

class DeepFusion(nn.Module):
    def __init__(self, image_model, text_model, hidden_dim=256, num_outputs=10):
        super(DeepFusion, self).__init__()
        
        self.image_model = image_model

        # t5
        # self.text_model = text_model.encoder
        
        # bert
        self.text_model = text_model

        # Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim
        text_feature_dim = self.text_model.embed_dim
        
        # Multi-modal fusion layer
        # text -> image
        self.t2i_attn = AttentionT2I(q_dim=image_feature_dim, # self.joint_embedding_size,
                                        k_dim=text_feature_dim,
                                        embed_dim=2048, # self.embed_dim,
                                        num_heads=8, # self.n_head,
                                        hidden_dim=1024, # self.t2i_hidden_dim,
                                        dropout=0.1,
                                        drop_path=.0,
                                        init_values=1.0 / 6, # cfg.MODEL.DYHEAD.NUM_CONVS,
                                        mode="t2i",
                                        use_layer_scale=True,
                                        clamp_min_for_underflow=True,
                                        clamp_max_for_overflow=True
                                        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs)
        )
        
    def forward(self, image, text):
        # Image feature extraction
        visual_features = self.image_model(image)

        print('image_features', visual_features.shape)
        
        # Text feature extraction
        language_feature = self.text_model(text).last_hidden_state[:, 0, :]  # [CLS] token

        print('text_features', language_feature.shape)
        
        # # Multi-modal fusion
        # q0, q1, q2, q3, q4 = self.t2i_attn(
        #             visual_features[0], visual_features[1],
        #             visual_features[2], visual_features[3],
        #             visual_features[4],
        #             language_feature, language_feature,
        #             attention_mask=text["attention_mask"]
        #         )

        # fused_visual_features = [q0, q1, q2, q3, q4]

        # Multi-modal fusion
        q = self.t2i_attn(
                    [visual_features],
                    language_feature, language_feature,
                    # attention_mask=text["attention_mask"]
                    attention_mask=None
                )
        
        print('q[0]', q[0].shape)

        fused_visual_features = q[0] + visual_features

        print('fused_visual_features', fused_visual_features.shape)
        
        # MLP
        x = self.mlp(fused_visual_features)

        output = {
            'left_fingertip': x[:, 0:3],
            'right_fingertip': x[:, 3:6],
            'force': x[:, 6:9],
            'pitch': x[:, 9]
        }

        return output