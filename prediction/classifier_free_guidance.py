#!/usr/bin/env python3

import torch
import torch.nn as nn
import timm
from transformers import T5Model, T5Tokenizer, BertModel, BertTokenizer
import clip
from utils.config_utils import *
from prediction.models import *
import numpy as np
from classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner, classifier_free_guidance

class ClassifierFreeGuidance(nn.Module):
    def __init__(self, image_model, text_model, hidden_dim=256, num_outputs=10):
        super(ClassifierFreeGuidance, self).__init__()
        
        self.image_model = image_model

        # self.text_model = text_model

        #  Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim
         
        # FiLM
        self.text_conditioner = TextConditioner(
            model_types = 't5',    
            hidden_dims = (1024,),
            hiddens_channel_first = False,
            cond_drop_prob = 0.2  # conditional dropout 20% of the time, must be greater than 0. to unlock classifier free guidance
        ).cuda()

        # # Cross-Attention
        # self.text_conditioner = AttentionTextConditioner(
        #     model_types = ('t5', 'clip'),   # something like in eDiff paper, where they used both T5 and Clip for even better results (Balaji et al.)
        #     # hidden_dims = tuple([image_feature_dim] * 24),
        #     hidden_dims = (1024,1024),
        #     cond_drop_prob = 0.2
        # ).cuda()

    @classifier_free_guidance # magic decorator
    def forward(self, image, cond_fns):
        # pass in your text as a List[str], and get back a List[callable]
        # each callable function receives the hiddens in the dimensions listed at init (hidden_dims)

        # first_condition_fn, second_condition_fn = self.text_conditioner([text])

        # # these hiddens will be in the direct flow of your model, say in a unet
        # first_hidden = torch.randn(1, 16, 256).cuda()
        # second_hidden = torch.randn(1, 32, 512).cuda()

        # # conditioned features
        # first_conditioned = first_condition_fn(first_hidden)
        # second_conditioned = second_condition_fn(second_hidden)

        image_features = self.image_model(image)

        print('intermediate layers: ', self.image_model.model.get_intermediate_layers(x=image,n=3))

        print('cond_fns', cond_fns)

        cond_fn = cond_fns[0] # get the first function in the list

        # Access the intermediate activations
        for idx, feature in enumerate(self.image_model.features.values()):
            print(f"Block {idx} output shape: {feature.shape}")
            cond_feature = cond_fn(feature).cuda() # condition the feature
            print(f"Block {idx} conditioned output shape: {cond_feature.shape}")
            self.image_model.features[idx] = cond_feature # replace the original feature with the conditioned feature so that the next block can use it


        x = torch.zeros((4, 10)).cuda()
        x.requires_grad = True

        output = {
            'left_fingertip': x[:, 0:3],
            'right_fingertip': x[:, 3:6],
            'force': x[:, 6:9],
            'pitch': x[:, 9]
        }

        return output  
    
class ConditionedVisionTransformer(nn.Module):
    def __init__(self, vit_model, text_model, config, hidden_dim=256, num_outputs=10, cond_method='xattn'):
        super(ConditionedVisionTransformer, self).__init__()
        self.vit_model = vit_model
        self.cond_method = cond_method
        self.config = config
        _, self.args = parse_config_args()

        if self.cond_method == 'film':
            # FiLM
            self.text_conditioner = TextConditioner(
                model_types = text_model, #'t5',    
                hidden_dims = tuple([self.vit_model.embed_dim] * len(self.vit_model.model.blocks)),
                hiddens_channel_first = False,
                cond_drop_prob = 0.0 # 0.2 # conditional dropout 20% of the time. Must be greater than 0 if you want classifier free guidance to work
            ).cuda()
        elif self.cond_method == 'xattn':
            # Cross-Attention
            self.text_conditioner = AttentionTextConditioner(
                # model_types = ('t5', 'clip'),   # something like in eDiff paper, where they used both T5 and Clip for even better results (Balaji et al.)
                model_types =  text_model,    
                hidden_dims = tuple([self.vit_model.embed_dim] * len(self.vit_model.model.blocks)),
                cond_drop_prob = 0. # 0.2
            ).cuda()

        # print('hidden_dims: ', tuple([self.vit_model.embed_dim] * len(self.vit_model.model.blocks)))

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.vit_model.embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_outputs)
        # )

        self.mlp_fingertips = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )

        self.mlp_force = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.mlp_grip_force = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
            self.classifier = nn.Sequential(
                nn.Linear(self.vit_model.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, config.NUM_TIMESTEPS),
                # nn.Softmax(dim=1) # TODO: remove since torch.nn.CrossEntropyLoss() already applies softmax
            )
        else:
            self.classifier = ClassificationHead(
                num_inputs=self.vit_model.embed_dim,
                num_outputs=config.NUM_TIMESTEPS)

        if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
            self.mlp_width = nn.Sequential(
                nn.Linear(self.vit_model.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
            self.mlp_yaw = nn.Sequential(
                nn.Linear(self.vit_model.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        # LINEAR DECODER
        # self.pixel_decoder = LinearDecoder(config, self.vit_model, patch_size=16, num_patches=196)
        # self.depth_decoder = LinearDecoder(config, self.vit_model, patch_size=16, num_patches=196)

        # CONV DECODER
        self.pixel_decoder = ConvDecoder(config, self.vit_model, patch_size=16, num_patches=196) # TODO: put decoders in config
        self.depth_decoder = ConvDecoder(config, self.vit_model, patch_size=16, num_patches=196)

    @classifier_free_guidance # magic decorator
    def forward(self, image, cond_fns): # model must take in "texts" as an arg when called, and cond_fns as an arg here
        # print(f"Input image shape: {image.shape}")
        # Preprocessing steps of the ViT model
        if self.config.USE_RGBD:
            x = self.vit_model.model.patch_embed(image) # patch embed input is RGBD
        else:
            x = self.vit_model.model.patch_embed(image[:, 0:3, :, :]) # patch embed input is RGB only

        # print(f"Patch embedding output shape: {x.shape}")
        x = self.vit_model.model._pos_embed(x) # includes pos_drop
        # print(f"pos embedding output shape: {x.shape}")
        # x = self.vit_model.model.patch_drop(x)
        # print(f"patch_drop output shape: {x.shape}")
        x = self.vit_model.model.norm_pre(x)
        # print(f"norm_pre output shape: {x.shape}")

        # Apply each block with conditioning
        for idx, (block, cond_fn) in enumerate(zip(self.vit_model.model.blocks, cond_fns)):
            # print(f"Block {idx}: {block}")
            x = block(x)
            # print(f"Block {idx} output shape: {x.shape}")
            if self.cond_method == 'film' and not self.args.ablate_prompt:
                x = cond_fn(x)
            elif self.cond_method == 'xattn' and not self.args.ablate_prompt:
                x = cond_fn(x.permute(0, 2, 1)).permute(0, 2, 1) # permuting for xattn then permuting back
            # print(f"Block {idx} conditioned output shape: {x.shape}")

        # Postprocessing steps of the ViT model
        x = self.vit_model.model.norm(x)
        # print(f"norm output shape: {x.shape}")
        x = self.vit_model.model.fc_norm(x)
        # print(f"fc_norm output shape: {x.shape}")
        # x = self.vit_model.model.head_drop(x) # TODO: REPLACE WITH DROPOUT
        x = self.vit_model.model.head(x) # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # print(f"head output shape: {x.shape}")

        # print('all features: ', x.shape)


        # avg pooled patch embeddings
        patch_feats = x[:, 1:]

        # cls token
        if self.config.PIXEL_SPACE_OUTPUT:
            cls = torch.mean(patch_feats, dim=1) # avg pool the patch embeddings instead of using the cls token. shape=(batch_size, embed_dim)
        else:
            cls = x[:, 0] # first element of the image features is a learned summary token. shape=(batch_size, embed_dim)
        # print('cls token: ', cls.shape)

        # print(f"cls token output shape: {x.shape}")

        # mlp_output = self.mlp(cls, texts=None)
        force = self.mlp_force(cls)
        grip_force = self.mlp_grip_force(cls)
        timestep = self.classifier(cls)

        if hasattr(self.config, 'PIXEL_SPACE_OUTPUT') and self.config.PIXEL_SPACE_OUTPUT:
            # output = torch.zeros((4, 2*224*224)).cuda()
            pixel_output = self.pixel_decoder(x[:, 1:]) # skip the cls token
            pixel_output = pixel_output.view(pixel_output.shape[0], -1)
            depth_output = self.depth_decoder(x[:, 1:]) # skip the cls token
            depth_output = depth_output.view(depth_output.shape[0], -1)
            output = torch.cat((pixel_output, depth_output), dim=1)
            # print(f'first {2*224*224} elements of output are images: ', output.shape)
        else: # fingertip xyz
            output = self.mlp_fingertips(cls)
            # print(f'first {fingertips.shape[1]} elements of output are fingertip xyz: ', output.shape)
        
        output = torch.cat((output, force), dim=1)
        # print(f'force is next {force.shape[1]} elements of output: ', output.shape)
        output = torch.cat((output, grip_force), dim=1)
        # print(f'grip force is next {grip_force.shape[1]} elements of output: ', output.shape)

        if hasattr(self.config, 'NUM_TIMESTEPS') and self.config.NUM_TIMESTEPS:       
            output = torch.cat((output, timestep), dim=1)
            # print(f'timestep is the next {timestep.shape[1]} elements of output: ', output.shape)

        if hasattr(self.config, 'LAMBDA_WIDTH') and self.config.LAMBDA_WIDTH:
            width = self.mlp_width(cls)
            output = torch.cat((output, width), dim=1)
            # print(f'width is the next {width.shape[1]} elements of output: ', output.shape)

        if hasattr(self.config, 'LAMBDA_YAW') and self.config.LAMBDA_YAW:
            yaw = self.mlp_yaw(cls)
            output = torch.cat((output, yaw), dim=1)

        # output = [fingertips, force, grip_force, timestep, width]
        return output  # classifier-free guidance library needs logits, not dict

class ConditionedCLIP(nn.Module):
    def __init__(self, clip_vit, text_model, config, hidden_dim=256, num_outputs=10, cond_method='xattn'):
        super(ConditionedCLIP, self).__init__()
        self.clip_vit = clip_vit
        self.cond_method = cond_method
        self.config = config
        _, self.args = parse_config_args()

        print('CLIP ViT!!!', self.clip_vit)

        if self.cond_method == 'film':
            # FiLM
            self.text_conditioner = TextConditioner(
                model_types = text_model, #'t5',    
                hidden_dims = tuple([self.clip_vit.embed_dim] * len(self.clip_vit.model.visual.transformer.resblocks)),
                hiddens_channel_first = False,
                cond_drop_prob = 0.0 # 0.2  # conditional dropout 20% of the time, must be greater than 0. to unlock classifier free guidance
            ).cuda()
        elif self.cond_method == 'xattn':
            # Cross-Attention
            self.text_conditioner = AttentionTextConditioner(
                # model_types = ('t5', 'clip'),   # something like in eDiff paper, where they used both T5 and Clip for even better results (Balaji et al.)
                model_types =  text_model,    
                hidden_dims = tuple([self.clip_vit.embed_dim] * len(self.clip_vit.model.visual.transformer.resblocks)),
                cond_drop_prob = 0. # 0.2
            ).cuda()

        # print('hidden_dims: ', tuple([self.clip_vit.embed_dim] * len(self.clip_vit.model.blocks)))

        self.mlp = nn.Sequential(
            nn.Linear(self.clip_vit.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs)
        )

        self.classifier = ClassificationHead(num_inputs=self.clip_vit.embed_dim, num_outputs=config.NUM_TIMESTEPS)

    @classifier_free_guidance # magic decorator
    def forward(self, image, cond_fns): # model must take in "texts" as an arg when called, and cond_fns as an arg here
        # print(f"Input image shape: {image.shape}")
        # Preprocessing steps of the ViT model
        x = self.clip_vit.model.visual.conv1(image)  # shape = [*, width, grid, grid]
        # print(f"conv1 output shape: {x.shape}")
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # print(f"reshape output shape: {x.shape}")
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # print(f"permute output shape: {x.shape}")
        x = torch.cat([self.clip_vit.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # print(f"cat output shape: {x.shape}")
        x = x + self.clip_vit.model.visual.positional_embedding.to(x.dtype)
        # print(f"positional_embedding output shape: {x.shape}")
        x = self.clip_vit.model.visual.ln_pre(x)
        # print(f"ln_pre output shape: {x.shape}")

        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(f"permute output shape: {x.shape}")
        
        # self.clip_vit.transformer.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        # iterating through the transformer blocks
        for idx, (block, cond_fn) in enumerate(zip(self.clip_vit.model.visual.transformer.resblocks, cond_fns)):
            # print(f"Block {idx}: {block}")
            x = block(x)
            # print(f"Block {idx} output shape: {x.shape}")
            if self.cond_method == 'film' and not self.args.ablate_prompt:
                x = cond_fn(x)
            elif self.cond_method == 'xattn' and not self.args.ablate_prompt:
                x = cond_fn(x.permute(1, 2, 0)).permute(2, 0, 1) # permuting for xattn then permuting back
                # print(f"xattn {idx} output shape: {x.shape}")

        # print(f"transformer output shape: {x.shape}")
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print(f"permute output shape: {x.shape}")

        x = self.clip_vit.model.visual.ln_post(x[:, 0, :])
        # print(f"ln_post output shape: {x.shape}")

        # if self.clip_vit.model.visual.proj is not None: # linear projection from 1024 to 768
        #     x = x @ self.clip_vit.model.visual.proj
        # print(f"proj output shape: {x.shape}")

        output = self.mlp(x)
        # print(f"mlp output shape: {output.shape}")

        output = torch.cat((output, self.classifier(x)), dim=1)

        return output  # classifier-free guidance library needs logits, not dict
    
class ConditionedDinov2(nn.Module):
    def __init__(self, vit_model, text_model, config, hidden_dim=256, num_outputs=10, cond_method='xattn'):
        super(ConditionedDinov2, self).__init__()
        self.vit_model = vit_model
        self.cond_method = cond_method
        self.config = config
        _, self.args = parse_config_args()

        if self.cond_method == 'film':
            # FiLM
            self.text_conditioner = TextConditioner(
                model_types = text_model, #'t5',    
                hidden_dims = tuple([self.vit_model.embed_dim] * len(self.vit_model.model.blocks)),
                hiddens_channel_first = False,
                cond_drop_prob = 0.0 # 0.2  # conditional dropout 20% of the time, must be greater than 0. to unlock classifier free guidance
            ).cuda()
        elif self.cond_method == 'xattn':
            # Cross-Attention
            self.text_conditioner = AttentionTextConditioner(
                # model_types = ('t5', 'clip'),   # something like in eDiff paper, where they used both T5 and Clip for even better results (Balaji et al.)
                model_types =  text_model,    
                hidden_dims = tuple([self.vit_model.embed_dim] * len(self.vit_model.model.blocks)),
                cond_drop_prob = 0. # 0.2
            ).cuda()

        # print('hidden_dims: ', tuple([self.vit_model.embed_dim] * len(self.vit_model.model.blocks)))

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.vit_model.embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_outputs)
        # )

        self.mlp_fingertips = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )

        self.mlp_force = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.mlp_grip_force = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
            self.classifier = nn.Sequential(
                nn.Linear(self.vit_model.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, config.NUM_TIMESTEPS),
                # nn.Softmax(dim=1) # TODO: remove since torch.nn.CrossEntropyLoss() already applies softmax
            )
        else:
            self.classifier = ClassificationHead(
                num_inputs=self.vit_model.embed_dim,
                num_outputs=config.NUM_TIMESTEPS)

        if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
            self.mlp_width = nn.Sequential(
                nn.Linear(self.vit_model.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
            self.mlp_yaw = nn.Sequential(
                nn.Linear(self.vit_model.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        # LINEAR DECODER
        # self.pixel_decoder = LinearDecoder(config, self.vit_model, patch_size=16, num_patches=196)
        # self.depth_decoder = LinearDecoder(config, self.vit_model, patch_size=16, num_patches=196)

        # CONV DECODER
        self.pixel_decoder = ConvDecoder(config, self.vit_model, patch_size=14, num_patches=256) # TODO: put decoders in config
        self.depth_decoder = ConvDecoder(config, self.vit_model, patch_size=14, num_patches=256)

    @classifier_free_guidance # magic decorator
    def forward(self, image, cond_fns): # model must take in "texts" as an arg when called, and cond_fns as an arg here
        # print('image shape: ', image.shape)
        x = self.vit_model.model.prepare_tokens_with_masks(image, masks=None)
        # print(f"prepare_tokens_with_masks output shape: {x.shape}")

        # Apply each block with conditioning
        for idx, (block, cond_fn) in enumerate(zip(self.vit_model.model.blocks, cond_fns)):
            # print(f"Block {idx}: {block}")
            x = block(x)
            # print(f"Block {idx} output shape: {x.shape}")
            if self.cond_method == 'film' and not self.args.ablate_prompt:
                x = cond_fn(x)
            elif self.cond_method == 'xattn' and not self.args.ablate_prompt:
                x = cond_fn(x.permute(0, 2, 1)).permute(0, 2, 1) # permuting for xattn then permuting back
            # print(f"Block {idx} conditioned output shape: {x.shape}")

        x = self.vit_model.model.norm(x)
        
        # return {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        #     "x_prenorm": x,
        #     "masks": None,
        # }

        # print('all features: ', x.shape)

        # avg pooled patch embeddings
        patch_feats = x[:, 1:]
        # print('patch feats: ', patch_feats.shape)

        # cls token
        if self.config.PIXEL_SPACE_OUTPUT:
            cls = torch.mean(patch_feats, dim=1) # avg pool the patch embeddings instead of using the cls token. shape=(batch_size, embed_dim)
        else:
            cls = x[:, 0] # first element of the image features is a learned summary token. shape=(batch_size, embed_dim)
        # print('cls token: ', cls.shape)

        # print(f"cls token output shape: {x.shape}")

        # mlp_output = self.mlp(cls, texts=None)
        force = self.mlp_force(cls)
        grip_force = self.mlp_grip_force(cls)
        timestep = self.classifier(cls)

        if hasattr(self.config, 'PIXEL_SPACE_OUTPUT') and self.config.PIXEL_SPACE_OUTPUT:
            # output = torch.zeros((4, 2*224*224)).cuda()
            pixel_output = self.pixel_decoder(x[:, 1:]) # skip the cls token
            pixel_output = pixel_output.view(pixel_output.shape[0], -1)
            depth_output = self.depth_decoder(x[:, 1:]) # skip the cls token
            depth_output = depth_output.view(depth_output.shape[0], -1)
            output = torch.cat((pixel_output, depth_output), dim=1)
            # print(f'first {2*224*224} elements of output are images: ', output.shape)
        else: # fingertip xyz
            output = self.mlp_fingertips(cls)
            # print(f'first {fingertips.shape[1]} elements of output are fingertip xyz: ', output.shape)
        
        output = torch.cat((output, force), dim=1)
        # print(f'force is next {force.shape[1]} elements of output: ', output.shape)
        output = torch.cat((output, grip_force), dim=1)
        # print(f'grip force is next {grip_force.shape[1]} elements of output: ', output.shape)

        if hasattr(self.config, 'NUM_TIMESTEPS') and self.config.NUM_TIMESTEPS:       
            output = torch.cat((output, timestep), dim=1)
            # print(f'timestep is the next {timestep.shape[1]} elements of output: ', output.shape)

        if hasattr(self.config, 'LAMBDA_WIDTH') and self.config.LAMBDA_WIDTH:
            width = self.mlp_width(cls)
            output = torch.cat((output, width), dim=1)
            # print(f'width is the next {width.shape[1]} elements of output: ', output.shape)

        if hasattr(self.config, 'LAMBDA_YAW') and self.config.LAMBDA_YAW:
            yaw = self.mlp_yaw(cls)
            output = torch.cat((output, yaw), dim=1)

        # output = [fingertips, force, grip_force, timestep, width]
        return output  # classifier-free guidance library needs logits, not dict
