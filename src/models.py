import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
        self.feature_extractor.classifier = nn.Identity()  # Remove the original classifier

    def extract_features(self, x):
        with torch.no_grad():
            return self.feature_extractor(x)

    def forward(self, x):
        return self.extract_features(x)

class BrainwaveEncoder(nn.Module):
    def __init__(self, seq_len, in_channels, out_feature_dim=1536): # ConvNeXt-Largeの出力特徴量は1536次元
        super().__init__()
        self.conv_classifier = ImprovedConvClassifier(out_feature_dim, seq_len, in_channels)

    def forward(self, x):
        return self.conv_classifier(x)

class CLIPModel(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels, out_feature_dim=1536):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.brainwave_encoder = BrainwaveEncoder(seq_len, in_channels, out_feature_dim)
        self.re_classifier = nn.Sequential(
            nn.Linear(out_feature_dim, num_classes),
            nn.Dropout(0.25),
            nn.Linear(num_classes, num_classes),
            nn.Dropout(0.25),
        )

    def forward(self, brainwaves, images=None):
        if images is not None:
            image_embeddings = self.image_encoder(images)
            brainwave_embeddings = self.brainwave_encoder(brainwaves)
            return image_embeddings, brainwave_embeddings
        else:
            brainwave_embeddings = self.brainwave_encoder(brainwaves)
            return self.re_classifier(brainwave_embeddings)

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Dropout(p=0.5),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)

# 改善された分類モデル
class ImprovedConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout_rate: float = 0.5
    ) -> None:
        super().__init__()

        # 畳み込みブロックを定義
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )
        
        # トランスフォーマーエンコーダー層の追加
        encoder_layers = TransformerEncoderLayer(hid_dim, nhead, hid_dim, dropout=dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # ヘッド部を定義（AdaptiveAvgPool1d, Dropout, Linear）
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)  # 畳み込みブロックを通過
        X = X.permute(2, 0, 1)  # (batch_size, seq_len, hid_dim) -> (seq_len, batch_size, hid_dim)
        X = self.transformer_encoder(X)  # トランスフォーマーエンコーダーを通過
        X = X.permute(1, 2, 0)  # (seq_len, batch_size, hid_dim) -> (batch_size, hid_dim, seq_len)
        return self.head(X)  # ヘッド部を通過

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)