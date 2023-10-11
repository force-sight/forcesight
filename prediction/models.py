import torch
import torch.nn as nn
import timm
from transformers import T5Model, T5EncoderModel, T5Tokenizer, BertModel, BertTokenizer
import clip
from utils.config_utils import *
import numpy as np
from einops import rearrange

##################################################################
# IMAGE ENCODERS
##################################################################

class RGBDViT(nn.Module):
    def __init__(self, size, config, num_classes=0, pretrained=True):
        super(RGBDViT, self).__init__()
        self.config = config
        print(f'Loading ViT-{size}...')
        print('Pretrained: ', pretrained)
            
        if size == 'tiny':
            self.model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
            self.embed_dim = 192
        elif size == 'small':
            self.model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
            self.embed_dim = 384
        elif size == 'base':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
            self.embed_dim = 768
        elif size == 'large':
            self.model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=num_classes) # ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
            # self.model = timm.create_model('vit_large_patch16_384', pretrained=pretrained, num_classes=num_classes) # ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
            self.embed_dim = 1024
        
        if self.config.USE_RGBD:
            # Modify the patch embedding layer to accept 4 channels (RGBD)
            original_patch_embedding = self.model.patch_embed.proj # (768, 3, 16, 16)
            self.model.patch_embed.proj = nn.Conv2d(4, self.model.embed_dim, kernel_size=(16, 16), stride=(16, 16), bias=False) # (768, 4, 16, 16). Conv with stride=16 is equivalent to having a linear transformation for each patch
            
            # Initialize the depth channel weights by averaging the weights from the RGB channels
            with torch.no_grad():
                self.model.patch_embed.proj.weight[:, :3] = original_patch_embedding.weight.clone()
                self.model.patch_embed.proj.weight[:, 3] = original_patch_embedding.weight.mean(dim=1) # Average the weights from the RGB channels

        # Register a forward hook for each block
        self.features = {}
        for idx, block in enumerate(self.model.blocks):
            block.register_forward_hook(self.hook)

        print('vit params: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def hook(self, module, input, output):
        self.features[module] = output

    def forward(self, x):
        self.features.clear()  # Clear the feature dictionary at the start of each forward pass
        if self.config.USE_RGBD:
            self.model.patch_embed.proj.requires_grad = True # always train the patch embedding layer
            return self.model(x)
        else:
            return self.model(x[:, :3])

class RGBDCLIP(nn.Module):
    def __init__(self, size, config, num_classes=0, pretrained=True):
        super(RGBDCLIP, self).__init__()
        self.config = config
        print(f'Loading CLIP-{size}...')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if size == 'base':
            self.model, self.preprocess = clip.load("ViT-B/32", device)
            self.embed_dim = 768
            patch_size = 32
        elif size == 'large':
            self.model, self.preprocess = clip.load("ViT-L/14", device)
            # self.model, self.preprocess = clip.load("ViT-L/14@336px", device)
            self.embed_dim = 1024
            patch_size = 14

        # converting all the weights from float16 to float32
        for p in self.model.parameters():
            p.data = p.data.float()

        if self.config.USE_RGBD:
            # Modify the patch embedding layer to accept 4 channels (RGBD)
            original_patch_embedding = self.model.visual.conv1
            self.model.visual.conv1 = nn.Conv2d(4, self.model.visual.conv1.out_channels, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=False) # equivalent to having a linear transformation for each patch

            # Initialize the depth channel weights by averaging the weights from the RGB channels
            with torch.no_grad():
                self.model.visual.conv1.weight[:, :3] = original_patch_embedding.weight.clone()
                self.model.visual.conv1.weight[:, 3] = original_patch_embedding.weight.mean(dim=1)
            

    def forward(self, x):
        # return self.model.encode_image(x), self.model.encode_text(x) # TODO: fix inputs
        if self.config.USE_RGBD:
            self.model.visual.conv1.requires_grad = True # always train the patch embedding layer for rgbd
            # return self.model.encode_image(x)
            return self.model.encode_image(x)
        else:
            image_features = self.model.encode_image(x[:, :3])
            return image_features
    
class RGBDDinov2(nn.Module):
    def __init__(self, size, config):
        super(RGBDDinov2, self).__init__()
        print(f'Loading DINOv2-{size}...')
        if size == 'small':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.embed_dim = 384
        elif size == 'base':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.embed_dim = 768
        elif size == 'large':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.embed_dim = 1024
        elif size == 'giant':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') # too big for rtx 3090
            self.embed_dim = 1536

        self.config = config
        if hasattr(config, 'USE_PATCH_FEATURES') and config.USE_PATCH_FEATURES:
            # self.embed_dim *= 257 # 256 patches + 1 cls token
            self.embed_dim *= 2 # avg pooled patch features + cls token

        if config.USE_RGBD:
            original_patch_embedding = self.model.patch_embed.proj
            self.model.patch_embed.proj = nn.Conv2d(4, self.model.patch_embed.proj.out_channels, kernel_size=(14, 14), stride=(14, 14), bias=False)

            with torch.no_grad():
                self.model.patch_embed.proj.weight[:, :3] = original_patch_embedding.weight.clone()
                self.model.patch_embed.proj.weight[:, 3] = original_patch_embedding.weight.mean(dim=1)
            

    def forward(self, x):
        if hasattr(self.config, 'USE_PATCH_FEATURES') and self.config.USE_PATCH_FEATURES:
            patch_features = self.model.forward_features(x[:, :3])
            # print('patch_features.keys:', patch_features.keys())
            # print('patch token shape:', patch_features['x_norm_patchtokens'].shape)
            # print('cls token shape:', patch_features['x_norm_clstoken'].unsqueeze(1).shape)
            # print('avg pool shape:', torch.mean(patch_features['x_norm_patchtokens'], dim=1).unsqueeze(1).shape)
            
            # return patch_features['x_norm_patchtokens'] # all patch tokens
            # return torch.cat((patch_features['x_norm_clstoken'].unsqueeze(1), patch_features['x_norm_patchtokens']), dim=1) # cls token + all patch tokens
            return torch.cat((patch_features['x_norm_clstoken'].unsqueeze(1), torch.mean(patch_features['x_norm_patchtokens'], dim=1).unsqueeze(1)), dim=1) # cls token + avg pool of patch tokens
        if self.config.USE_RGBD:
            self.model.patch_embed.proj.requires_grad = True # always train the patch embedding layer for rgbd
            return self.model(x)
        else:
            return self.model(x[:, :3])


##################################################################
# IMAGE DECODERS
##################################################################

class LinearPatchDecoder(nn.Module):
    # takes in a batch of patch embeddings and outputs a batch of images
    def __init__(self, image_model, patch_size, num_channels=1):
        super(LinearPatchDecoder, self).__init__()
        self.image_model = image_model
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.linear = nn.Linear(image_model.embed_dim, patch_size * patch_size * num_channels)

    def forward(self, x):
        # print('LinearPatchDecoder input:', x.shape)
        x = self.linear(x)
        # print('LinearPatchDecoder output:', x.shape)
        x = x.view(-1, self.num_channels, self.patch_size, self.patch_size)
        # print('Reshaped LinearPatchDecoder output:', x.shape)
        return x

class LinearDecoder(nn.Module):
    # takes in a batch of patch embeddings and outputs a batch of images
    def __init__(self, config, image_model, patch_size, num_patches, num_channels=1):
        super(LinearDecoder, self).__init__()
        self.config = config
        self.image_model = image_model
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_patches_side = int(np.sqrt(num_patches)) # assuming square image

        # making a LinearPatchDecoder for each patch
        self.linear = nn.ModuleList([nn.Linear(image_model.embed_dim, patch_size * patch_size * num_channels) for _ in range(num_patches)])

    def forward(self, x):
        # print('LinearDecoder input:', x.shape)
        x = torch.stack([linear(x[:, i, :]) for i, linear in enumerate(self.linear)], dim=1)
        # print('LinearDecoder output:', x.shape) # (batch_size, num_patches, num_channels * patch_size * patch_size)
        
        # reshaping to (batch_size, num_patches, num_channels, patch_size, patch_size)
        x = x.view(-1, self.num_patches, self.num_channels, self.patch_size, self.patch_size)

        # reshaping to the original image size
        # x = x.view(-1, self.num_channels, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)

        # flattening to a vector
        # x = x.view(-1, self.num_patches * self.num_channels * self.patch_size * self.patch_size)
        # print('Reshaped LinearDecoder output:', x.shape)

        # Rearrange the patches to (batch_size, num_channels, image_size, image_size)
        x = rearrange(x, 'b (h w) c ph pw -> b c (h ph) (w pw)', h=self.num_patches_side, w=self.num_patches_side, ph=self.patch_size, pw=self.patch_size)

        # print('Rearranged LinearDecoder output:', x.shape) (batch_size, num_channels, image_size, image_size)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, config, image_model, patch_size, num_patches, num_channels=1):
        # takes in a batch of patch embeddings (196, 1024) and outputs a batch of images (1, 224, 224)
        super(ConvDecoder, self).__init__()
        self.config = config
        self.image_model = image_model

        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.image_model.embed_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # 14 x 14 x 512
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # 28 x 28 x 256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 56 x 56 x 1
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 112 x 112 x 1
        )

    def forward(self, x):
        # x.shape will be (batch_size, 196, 1024)
        x = x.transpose(1, 2)  # swap the patch and embedding dimensions
        # x.shape is now (batch_size, 1024, 196)
        
        B, C, N = x.shape
        H = W = int(N ** 0.5)  # assumes number of patches is a perfect square

        x = x.view(B, C, H, W)  # rearrange patches to 2D grid
        # x.shape is now (batch_size, 1024, 14, 14)

        # upscale to 224x224
        x = self.up(x) # (batch_size, 1, 224, 224)
        return x  

##################################################################
# TEXT ENCODERS
##################################################################

class Bert(nn.Module):
    def __init__(self, size):
        super(Bert, self).__init__()
        print(f'Loading BERT-{size}...')
        if size == 'base':
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.embed_dim = 768
        elif size == 'large':
            self.model = BertModel.from_pretrained('bert-large-uncased')
            self.embed_dim = 1024

    def forward(self, x):
        # Extract the required tensors
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        return self.model(input_ids, attention_mask).last_hidden_state[:, 0, :] # odict_keys(['last_hidden_state', 'pooler_output'])

class T5(nn.Module):
    def __init__(self, size):
        super(T5, self).__init__()
        print(f'Loading T5-{size}...')
        if size == 'small':
            self.model = T5Model.from_pretrained("t5-small").encoder
            self.embed_dim = 512
        elif size == 'base':
            self.model = T5Model.from_pretrained("t5-base").encoder
            self.embed_dim = 768
        elif size == 'large':
            self.model = T5Model.from_pretrained("t5-large").encoder
            self.embed_dim = 1024
        # elif size == '3b':
        #     self.model = T5EncoderModel.from_pretrained('t5-3b')
        #     self.embed_dim = 1024
        # elif size == '11b':
        #     self.model = T5EncoderModel.from_pretrained('t5-11b')
        #     self.embed_dim = 1024

        print(f'T5 num params:', {sum(p.numel() for p in self.model.parameters())})
    
    def forward(self, x):
        return self.model(x).last_hidden_state[:, 0, :] # CLS token

##################################################################
# MULTIMODAL HEADS
##################################################################

class ConcatLinearAttnMLP(nn.Module):
    def __init__(self, image_model, text_model, hidden_dim=256, num_outputs=10):
        super(ConcatLinearAttnMLP, self).__init__()
        
        self.image_model = image_model
        self.text_model = text_model
        
        # Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim

        # t5
        # text_feature_dim = self.text_model.config.d_model

        # bert
        # text_feature_dim = self.text_model.embed_dim

        # clip
        # text_feature_dim = 768

        text_feature_dim = self.text_model.embed_dim
        
        # Multi-modal fusion layer
        self.fusion_layer = nn.Linear(image_feature_dim + text_feature_dim, hidden_dim)
        # self.fusion_layer = nn.Linear(image_feature_dim, hidden_dim)
        
        # Attention layers
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        
    def forward(self, image, texts):
        # Image feature extraction
        image_features = self.image_model(image)

        # Text feature extraction
        text_features = self.text_model(texts)
        # text_features = text_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Multi-modal fusion
        joint_features = torch.cat((image_features, text_features), dim=1)
        # joint_features = image_features

        # print('concatenated features', joint_features.shape)

        joint_features = self.fusion_layer(joint_features)

        # print('joint_features', joint_features.shape)
        
        # Apply attention
        joint_features = joint_features.unsqueeze(0)
        attended_features, _ = self.attention(joint_features, joint_features, joint_features) # ignore attention weights
        attended_features = self.attention_norm(attended_features + joint_features)
        attended_features = attended_features.squeeze(0)

        # print('attended_features', attended_features.shape)
        
        # Prediction layers
        x = self.fc1(attended_features)

        # print('fc1', x.shape)

        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        # print('fc2', x.shape)
        
        output = {
            'left_fingertip': x[:, 0:3],
            'right_fingertip': x[:, 3:6],
            'force': x[:, 6:9],
            'grip_force': x[:, 9]
        }

        return output
    
class VisionOnlyConcatLinearAttnMLP(nn.Module):
    def __init__(self, image_model, text_model, hidden_dim=256, num_outputs=10):
        super(ConcatLinearAttnMLP, self).__init__()
        
        self.image_model = image_model
        self.text_model = text_model
        
        # Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim
        
        # Attention layers
        self.attention = nn.MultiheadAttention(image_feature_dim, num_heads=8)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        
    def forward(self, image, texts):
        # Image feature extraction
        image_features = self.image_model(image)

        joint_features = image_features

        # Apply attention
        joint_features = joint_features.unsqueeze(0)
        attended_features, _ = self.attention(joint_features, joint_features, joint_features) # ignore attention weights
        attended_features = self.attention_norm(attended_features + joint_features)
        attended_features = attended_features.squeeze(0)

        # Prediction layers
        x = self.fc1(attended_features)
        x = torch.relu(x)
        x = self.fc2(x)
        
        output = {
            'left_fingertip': x[:, 0:3],
            'right_fingertip': x[:, 3:6],
            'force': x[:, 6:9],
            'grip_force': x[:, 9]
        }

        return output
    
class VisionOnlyLinear(nn.Module):
    def __init__(self, image_model, num_outputs=10):
        super(VisionOnlyLinear, self).__init__()
        # vision model with a single linear layer at the end
        
        self.image_model = image_model
        
        # Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim
        
        # Linear layer
        self.fc1 = nn.Linear(image_feature_dim, num_outputs)
        self.fc1.weight.data.normal_(mean=0.0, std=0.01) # from dinov2 code
        self.fc1.bias.data.zero_() # from dinov2 code
        
    def forward(self, image, texts):
        # Image feature extraction
        image_features = self.image_model(image)

        # print('image_features', image_features.shape)

        # making sure the image features are 1D
        image_features = image_features.view(image_features.size(0), -1)
        
        # Linear layer
        x = self.fc1(image_features)
        
        output = {
            'left_fingertip': x[:, 0:3],
            'right_fingertip': x[:, 3:6],
            'force': x[:, 6:9],
            'grip_force': x[:, 9]
        }

        return output
    
class VisionOnlyMLP(nn.Module):
    def __init__(self, image_model, hidden_dim=1024, num_outputs=10):
        super(VisionOnlyMLP, self).__init__()
        # vision model with an MLP at the end
        
        self.image_model = image_model
        
        # Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim
        
       # MLP
        self.fc1 = nn.Linear(image_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        
    def forward(self, image, texts):
        # Image feature extraction
        image_features = self.image_model(image)

        # making sure the image features are 1D
        image_features = image_features.view(image_features.size(0), -1)

        # MLP
        x = self.fc1(image_features)
        x = torch.relu(x)
        x = self.fc2(x)
        
        output = {
            'left_fingertip': x[:, 0:3],
            'right_fingertip': x[:, 3:6],
            'force': x[:, 6:9],
            'grip_force': x[:, 9]
        }

        return output
    
class ThreeHeadMLP(nn.Module):
    def __init__(self, image_model, hidden_dim=1024, num_outputs=10):
        super(ThreeHeadMLP, self).__init__()
        # vision model with an MLP at the end
        
        self.image_model = image_model
        
        # Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim
        
       # MLP
        # self.fc1 = nn.Linear(image_feature_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, num_outputs)

        self.mlp_fingertips = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )

        self.mlp_force = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.mlp_grip_force = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, image, texts):
        # Image feature extraction
        image_features = self.image_model(image)

        # making sure the image features are 1D
        image_features = image_features.view(image_features.size(0), -1)
        
        fingertips = self.mlp_fingertips(image_features)
        force = self.mlp_force(image_features)
        grip_force = self.mlp_grip_force(image_features)
        
        output = {
            'left_fingertip': fingertips[:, 0:3],
            'right_fingertip': fingertips[:, 3:6],
            'force': force,
            'grip_force': grip_force
        }

        return output
    
class ClassificationHead(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=1024):
        super(ClassificationHead, self).__init__()
        
       # MLP
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        
    def forward(self, x):

        # MLP
        x = self.fc1(x)
        x = torch.relu(x)
        output = self.fc2(x)
        
        # output = torch.softmax(x, dim=1) # TODO: remove since torch.nn.CrossEntropyLoss() already applies softmax

        return output

class PixelClassifier(nn.Module):
    def __init__(self, image_model, image_size=224):
        super(PixelClassifier, self).__init__()
        # linear classifier for each pixel
        self.image_size = image_size
        self.linear = nn.Linear(image_model.embed_dim, image_size**2)

    def forward(self, image_features):
        # making sure the image features are 1D
        image_features = image_features.view(image_features.size(0), -1)
        output = self.linear(image_features)
        # reshape to image size
        output = output.view(output.size(0), self.image_size, self.image_size)
        return output

if __name__ == "__main__":
    # Instantiate the pre-trained models
    config, args = parse_config_args()
    image_model = RGBDViT(num_classes=0, config=config)
    text_model = T5Model.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Instantiate the multi-modal model
    model = ConcatLinearAttnMLP(image_model, text_model)

    # Test the model
    x_image = torch.randn(1, 4, config.IMAGE_SIZE, config.IMAGE_SIZE)
    x_text = ["turn left"]

    # Add padding to the text input
    text_data_padded = tokenizer.batch_encode_plus(
        x_text,
        padding='max_length',
        max_length=32,
        return_tensors='pt',
        truncation=True
    )["input_ids"]

    # Forward pass
    y = model(x_image, text_data_padded)

    print('Output shape:', y.shape)  # [batch_size, num_outputs]