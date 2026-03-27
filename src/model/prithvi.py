from enum import StrEnum
import torch
import torch.nn as nn
import terratorch
from terratorch import BACKBONE_REGISTRY
from terratorch.models.necks import SelectIndices, ReshapeTokensToImage, LearnedInterpolateToPyramidal
from terratorch.models.decoders import LinearDecoder, FCNDecoder, UperNetDecoder
from terratorch.datasets import HLSBands
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from terratorch.models.backbones.prithvi_vit import PrithviViT

from models.central_block.memory_block import MemoryBlock

class PrithviDecoder(StrEnum):
    FCNDecoder = "FCNDecoder"
    FPNDecoder = "FPNDecoder"
    LinearDecoder = "LinearDecoder"
    UpperNetDecoder = "UpperNetDecoder"

class PrithviModel(torch.nn.Module):
    DIM = 192
    
    def __init__(self,
                 n_classes:int =2,
                 decoder: str = PrithviDecoder.LinearDecoder,
                 embed_tokens: int = 16*16,
                 embed_dim: int = 192,
                 fuse_stage: int = 11,
                 l2_norm: bool = True,
                 baseline: bool = False,
                 num_frames: int = 1,
                 image_size: int = 256,
                 **kwargs
                 ):
        super().__init__()

        encoder_channels = [48,48,96,192,192]
        self.num_frames = 1
        self.embed_dim = embed_dim
        self.baseline = baseline

        if self.baseline:
            self.num_frames = num_frames
            encoder_channels = [c*self.num_frames for c in encoder_channels]
            self.encoder = BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl", pretrained=True, num_frames=self.num_frames)
        else:
            self.encoder = create_prithvi_tiny(fuse_stage=fuse_stage, mem_dim=embed_dim)
            
        self.embed_tokens = embed_tokens
        self.em_shape = (embed_tokens, self.DIM)
        self.use_l2_norm = l2_norm
        self.n_classes = n_classes
        
        decoder_dim = 128

        self.decoder_type = decoder

        match decoder:
            case PrithviDecoder.LinearDecoder:
                self.neck = nn.Sequential(
                    SelectIndices(indices=[2,5,8,11], channel_list=[self.DIM]*12),
                    ReshapeTokensToImage(channel_list=[self.DIM]*4, remove_cls_token=True, effective_time_dim=self.num_frames),
                )
                self.decoder = LinearDecoder(embed_dim=[self.DIM*self.num_frames], num_classes=n_classes, upsampling_size=16, in_index=-1)
            case PrithviDecoder.FCNDecoder:
                self.neck = nn.Sequential(
                    SelectIndices(indices=[-1], channel_list=[self.DIM]*12),
                    ReshapeTokensToImage(channel_list=[self.DIM], remove_cls_token=True, effective_time_dim=self.num_frames),
                )
                self.decoder = FCNDecoder(embed_dim=[self.DIM*self.num_frames], channels=decoder_dim, num_convs=4, in_index=-1)
            case PrithviDecoder.FPNDecoder:
                self.neck = nn.Sequential(
                    SelectIndices(indices=[2,5,8,11], channel_list=[self.DIM]*12),
                    ReshapeTokensToImage(channel_list=[self.DIM], remove_cls_token=True, effective_time_dim=self.num_frames),
                    LearnedInterpolateToPyramidal(channel_list=[self.DIM*self.num_frames]*4)
                )
                self.decoder = CustomFPNDecoder(encoder_channels=encoder_channels, decoder_dim=decoder_dim)
            case PrithviDecoder.UpperNetDecoder:
                self.neck = nn.Sequential(
                    SelectIndices(indices=[2,5,8,11], channel_list=[self.DIM]*12),
                    ReshapeTokensToImage(channel_list=[self.DIM]*4, remove_cls_token=True, effective_time_dim=self.num_frames),
                    LearnedInterpolateToPyramidal(channel_list=[self.DIM*self.num_frames]*4)
                )
                self.decoder = nn.Sequential(
                    UperNetDecoder(embed_dim = encoder_channels[-4:], channels=decoder_dim),
                    nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=4, stride=4, bias=False),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True)
                )
            case _:
                raise ValueError(f"Decoder {decoder} not recognized.")
                
        self.classification_head = nn.Linear(self.DIM, n_classes)

        if self.decoder_type == PrithviDecoder.LinearDecoder:
            self.segmentation_head = nn.Identity()
        else:
            self.segmentation_head = nn.Conv2d(128, n_classes, kernel_size=1)
        
        if self.baseline:
            self.forward = self._forward_baseline
        
    def _l2_norm(self, x):
        return x / torch.norm(x, dim=2, keepdim=True)

    def _get_features_and_token(self, x, em):

        B = x.shape[0]
        
        if em is None and self.embed_dim > 0:
            em = torch.zeros((B, self.embed_tokens, self.embed_dim), dtype=x.dtype, device=x.device)
        elif em is not None and len(em.shape) == 4:
            em = em.flatten(2).transpose(1,2)

        features, em_out = self.encoder(x, em)
            
        cls_token = features[-1][:, 0, :]
        features = self.neck(features)
        return features, cls_token, em_out
    
    def _forward_baseline(self, before, after):
        B,C,H,W = after.shape
            
        if len(before) != self.num_frames - 1:
            print("WARNING: strange number of before images", len(before))
            return torch.zeros(B, self.n_classes, H, W, device=after.device)
        
        imgs = before + [after]
        B,C,H,W = after.shape
        imgs = torch.cat(imgs, dim=1)
        imgs = imgs.view(B,C,-1,H,W)

        x = self.encoder(imgs)
        x = self.neck(x)
        x = self.decoder(x)
        out = self.segmentation_head(x)
        return out

    def forward(self, x, em, return_cls=False, cls_per_token=False):
        B = x.shape[0]

        features, cls_token, new_em = self._get_features_and_token(x, em)

        # L2 normalization per token
        if self.use_l2_norm:
            new_em = new_em / torch.norm(new_em, dim=2, keepdim=True)

        out = self.decoder(features)

        out = self.segmentation_head(out)

        pred_tokens = None
        if cls_per_token:
            N = int(self.embed_tokens**0.5)
            pred_tokens = self.classification_head(new_em.view(-1, self.DIM)).view(B, self.embed_tokens, self.n_classes)
            pred_tokens = pred_tokens.permute(0,2,1).view(B, self.n_classes, N, N)
            return pred_tokens, out, new_em
        
        if return_cls: 
            pred = self.classification_head(cls_token) 
            return pred, out, new_em
        
        return out, new_em


    def forward_series(self, before, after, return_cls=False):

        assert not self.baseline, "Baseline model does not support series forward. Use _forward_baseline instead."
         
        outs, preds, ems = [], [], []
        
        em = None
        for img in before + [after]:
            pred, out, em = self.forward(img, em, return_cls=True)
            outs.append(out)
            preds.append(pred)
            ems.append(em.clone() if em is not None else None)

        if return_cls:
            return preds, outs, ems
        
        return outs, ems

    def forward_series_train(self, before, after, em=None):

        B = after.shape[0]
        
        if em is None and self.embed_dim > 0:
            em = torch.zeros((B, self.embed_tokens, self.embed_dim), dtype=after.dtype, device=after.device)
        
        early_outs = []
        _, em = self.forward(before[0], em, return_cls=False)
        outs = []
        ems  = [em.clone() if em is not None else None]
        
        for img in before[1:]:
            stacked = torch.cat([img, after], dim=0)
            em_stacked = torch.cat([em, em], dim=0) if em is not None else None
            out, em_stacked = self.forward(stacked, em_stacked, return_cls=False)
            em = em_stacked[:B] if em is not None else None
            outs.append(out[:B])
            ems.append(em.clone() if em is not None else None)
            early_outs.append((out[B:]))

        out, em = self.forward(after, em, return_cls=False)
        outs.append(out)
        ems.append(em.clone() if em is not None else None)

        return outs, ems, early_outs


class CustomFPNDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_dim=128, pyramid_channels=256):
        super().__init__()
        self.decoder = FPNDecoder(encoder_channels=encoder_channels, pyramid_channels=256, segmentation_channels=decoder_dim)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, features):
        x = self.decoder(*features)
        x = self.up(x)
        return x


class CustomPrithViEncoder(PrithviViT):

    def __init__(self, fuse_stage:int=0, mem_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.fuse_stage = fuse_stage
        self.mem_dim = mem_dim

        if mem_dim > 0:
            self.mem_proj = nn.Linear(self.mem_dim, 192) if self.mem_dim !=192 else nn.Identity()
            self.adapter = nn.Sequential(
                nn.LayerNorm(192),
                nn.Linear(192, self.mem_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        embed: None | torch.Tensor = None,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
        sample_shape = x.shape[-3:]

        B = x.shape[0]

        # embed patches
        x = self.patch_embed(x)


        pos_embed = self.interpolate_pos_encoding(sample_shape)
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Custom embeddings
        E = 0
        if embed is not None:
            # assert embed.shape[-1] == cls_token.shape[-1], f"Embedding dim {embed.shape} does not match model {cls_token.shape[-1]}"
            E = embed.shape[1]
            embed = self.mem_proj(embed)
            step = (pos_embed.shape[1] - 1) // embed.shape[1]
            embed = embed + pos_embed[:, step // 2 +1 ::step, :]
            # x = torch.cat((embed, x), dim=1) 

        # apply Transformer blocks
        bs = x.shape[0]
        features = []
        for idx, block in enumerate(self.blocks):
            if idx == self.fuse_stage and embed is not None:
                x = torch.cat((
                    x[:, :1, :],
                    embed,
                    x[:, 1:, :],
                ), dim=1)
            if self.vpt:
                x = torch.cat(
                    (
                        x[:, :1, :],
                        self.vpt_dropout_layers[idx](self.vpt_prompt_embeddings[idx].expand(bs, -1, -1)),
                        x[:, 1:, :],
                    ),
                    dim=1,
                )  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
            x = block(x)
            if self.vpt:
                x = torch.cat(
                    (x[:, :1, :], x[:, (1 + self.vpt_n_tokens) :, :]),
                    dim=1,
                )
            if idx == self.fuse_stage and embed is not None:
                embed = x[:, 1:1+E, :].clone()
                embed = self.adapter(embed)
                x = torch.cat(
                    (x[:, :1, :], x[:, (1 + E) :, :]),
                    dim=1,
                )

             
            # Collect features without Embed tokens
            features.append(x.clone())

        features[-1] = self.norm(x)

        return features, embed

def create_prithvi_tiny(fuse_stage=11, mem_dim=192):
    m = CustomPrithViEncoder(
        fuse_stage=fuse_stage, mem_dim=mem_dim, 
        num_frames=1, embed_dim=192, depth=12, num_heads=3, decoder_embed_dim=512, decoder_depth=8,
        decoder_num_heads=16, coords_encoding=["time", "location"], coords_scale_learn=True,in_chans=6
    )
    PRETRAINED_BANDS = [
        HLSBands.BLUE,
        HLSBands.GREEN,
        HLSBands.RED,
        HLSBands.NIR_NARROW,
        HLSBands.SWIR_1,
        HLSBands.SWIR_2,
    ]
    m.load_state_dict(BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl", bands=PRETRAINED_BANDS).state_dict(), strict=False)
    return m


