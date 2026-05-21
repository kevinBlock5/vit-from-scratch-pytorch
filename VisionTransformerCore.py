# Importing dependencies...
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# Image patching and embedding Layer...
class PatchEmbedding(nn.Module):
    """
    Splits an image into patches, projects them into a vector space,
    adds positional embeddings, and applies dropout.
    
    Supports dynamic spatial resolution scaling via interpolation.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, in_channels, height, width)
        batch_size = x.shape[0]
        out = self.patcher(x)  # shape: (batch_size, embed_dim, grid_h, grid_w)
        grid_h, grid_w = out.shape[2], out.shape[3]
        
        out = out.flatten(2).transpose(1, 2)  # shape: (batch_size, grid_h * grid_w, embed_dim)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)  # shape: (batch_size, grid_h * grid_w + 1, embed_dim)
        
        # Handle dynamic positional embedding interpolation if spatial resolution changes
        pos_embed = self.pos_embedding
        if out.shape[1] != pos_embed.shape[1]:
            cls_pos_embed = pos_embed[:, :1, :]
            spatial_pos_embed = pos_embed[:, 1:, :]
            
            orig_h = orig_w = int(self.num_patches ** 0.5)
            # Reshape to (1, embed_dim, orig_h, orig_w) for interpolation
            spatial_pos_embed = spatial_pos_embed.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)
            spatial_pos_embed = F.interpolate(
                spatial_pos_embed,
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False
            )
            # Reshape back to (1, grid_h * grid_w, embed_dim)
            spatial_pos_embed = spatial_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat((cls_pos_embed, spatial_pos_embed), dim=1)
            
        out = out + pos_embed
        out = self.dropout(out)
        return out


# EncoderStack...
class EncoderStack(nn.Module):
    """A stack of standard Transformer Encoder layers."""
    def __init__(
        self,
        nheads: int,
        embed_dim: int,
        layers: int,
        dropout: float,
        batch_first: bool
    ) -> None:
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nheads,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=True,
            activation="gelu"
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x)
    

# Classification Head...
class Head(nn.Module):
    """
    Classification head for the Vision Transformer.
    Supports either a simple linear projection or a multi-layer perceptron (MLP).
    """
    def __init__(
        self,
        embed_dim: int,
        classes: int,
        representation_size: Optional[int] = None
    ) -> None:
        super().__init__()
        if representation_size is not None:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh(),
                nn.Linear(representation_size, classes)
            )
        else:
            self.head = nn.Linear(embed_dim, classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# VisionTransformer...
class VisionTransformer(nn.Module):
    """
    Standard Vision Transformer (ViT) implementation.
    Supports input size variations, customizable heads, and proper seq-first configurations.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        nheads: int,
        layers: int,
        dropout: float,
        classes: int,
        batch_first: bool = True,
        representation_size: Optional[int] = None
    ) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout
        )
        self.encoder = EncoderStack(
            nheads=nheads,
            embed_dim=embed_dim,
            layers=layers,
            dropout=dropout,
            batch_first=batch_first
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.head = Head(
            embed_dim=embed_dim,
            classes=classes,
            representation_size=representation_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding is always batch-first initially
        out = self.embedding(x)
        
        # Transpose if the encoder expects sequence-first tensors
        if not self.batch_first:
            out = out.transpose(0, 1)
            
        out = self.encoder(out)
        
        # Transpose back to batch-first to extract the class token
        if not self.batch_first:
            out = out.transpose(0, 1)

        cls_token_out = out[:, 0, :]
        cls_token_out = self.ln(cls_token_out)
        out = self.head(cls_token_out)

        return out


if __name__ == "__main__":
    # Test 1: Standard input with batch_first=True
    print("Running smoke tests for VisionTransformer...")
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        nheads=3,
        layers=4,
        dropout=0.1,
        classes=10,
        batch_first=True
    )
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Test 1 (Standard) - Output shape:", tuple(output.shape))
    assert output.shape == (2, 10), "Test 1 shape mismatch!"

    # Test 2: Standard input with batch_first=False
    model_seq_first = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        nheads=3,
        layers=4,
        dropout=0.1,
        classes=10,
        batch_first=False
    )
    output_seq_first = model_seq_first(dummy_input)
    print("Test 2 (Seq-first) - Output shape:", tuple(output_seq_first.shape))
    assert output_seq_first.shape == (2, 10), "Test 2 shape mismatch!"

    # Test 3: Dynamic image size (interpolation verification)
    dummy_input_large = torch.randn(2, 3, 256, 256)
    output_large = model(dummy_input_large)
    print("Test 3 (Dynamic resolution 256x256) - Output shape:", tuple(output_large.shape))
    assert output_large.shape == (2, 10), "Test 3 shape mismatch!"
    
    print("All tests passed successfully!")