import torch
import torch.nn as nn
from timm.models import create_model

class VPTViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vit = create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)

        # Freeze all except prompt and head
        for param in self.vit.parameters():
            param.requires_grad = False

        embed_dim = self.vit.embed_dim
        self.prompt = nn.Parameter(torch.randn(1, cfg.prompt_length, embed_dim))
        self.prompt_length = cfg.prompt_length

        self.vit.head = nn.Linear(embed_dim, cfg.num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(B, -1, -1)
        prompt = self.prompt.expand(B, -1, -1)
        x = torch.cat((cls_token, prompt, x), dim=1)

        pos_embed = self.vit.pos_embed[:, :(x.size(1) + 1), :]
        x = self.vit.pos_drop(x + pos_embed)

        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)
        return self.vit.head(x[:, 0])