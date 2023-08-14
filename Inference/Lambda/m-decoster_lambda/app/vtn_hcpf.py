import torch
from torch import nn

from common import FeatureExtractor, LinearClassifier, SelfAttention


class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std


class VTNHCPF(nn.Module):
    def __init__(self, num_classes=226, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()

        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)
        num_attn_features = 2 * embed_size
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(66 + num_attn_features, num_attn_features)

        self.self_attention_decoder = SelfAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers,
                                                    self.sequence_length, layer_norm=True)
        self.classifier = LinearClassifier(num_attn_features, num_classes, dropout)

    def forward(self, mm_clip):
        """Extract the image feature vectors."""
        rgb_clip, pose_clip = mm_clip

        # Reshape to put both hand crops on the same axis.
        b, t, x, c, h, w = rgb_clip.size()

        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        z = self.feature_extractor(rgb_clip)
        # Reshape back to extract features of both wrist crops as one feature vector.
        z = z.view(b, t, -1)

        zp = torch.cat((z, pose_clip), dim=-1)

        zp = self.norm(zp)
        zp = torch.nn.functional.relu(self.bottle_mm(zp))

        zp = self.self_attention_decoder(zp)

        y = self.classifier(zp)

        return y.mean(1)
