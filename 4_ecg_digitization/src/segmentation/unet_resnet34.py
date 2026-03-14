import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet34UNet(nn.Module):
    """
    Faza 2: Semantic Segmentation ze świadomością nakładania.
    Enkoder oparty na architekturze ResNet34 (duże pole recepcyjne)
    pozwalający na zrozumienie szerokiego kontekstu krzywej EKG,
    aby odrzucić linie przecinające się (Overlay).
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(ResNet34UNet, self).__init__()
        
        # Enkoder z pre-trenowanymi wagami z ImageNet
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        
        # Jeśli obraz wejściowy jest grayscale, modyfikujemy pierwszą warstwę konwolucyjną
        if in_channels != 3:
            self.encoder0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Uśredniamy wagi z 3 kanałów do in_channels
            self.encoder0.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True)
        else:
            self.encoder0 = resnet.conv1
            
        self.encoder0_bn = resnet.bn1
        self.encoder0_relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1 # Wyjście: 64 kanały
        self.encoder2 = resnet.layer2 # Wyjście: 128 kanałów
        self.encoder3 = resnet.layer3 # Wyjście: 256 kanałów
        self.encoder4 = resnet.layer4 # Wyjście: 512 kanałów
        
        # Dekoder (Up-Sampling + Skip Connections)
        self.upconv4 = self._upconv(512, 256)
        self.decoder4 = self._conv_block(256 + 256, 256)
        
        self.upconv3 = self._upconv(256, 128)
        self.decoder3 = self._conv_block(128 + 128, 128)
        
        self.upconv2 = self._upconv(128, 64)
        self.decoder2 = self._conv_block(64 + 64, 64)
        
        self.upconv1 = self._upconv(64, 64)
        self.decoder1 = self._conv_block(64 + 64, 64)
        
        self.upconv0 = self._upconv(64, 32)
        self.decoder0 = self._conv_block(32, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def _upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _pad_to_match(self, x, target):
        """Paddowanie wyjścia w przypadku nierównych wymiarów obrazu po maxpoolingu"""
        diffY = target.size()[2] - x.size()[2]
        diffX = target.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, x):
        # Ścieżka kodowania (Encoder)
        e0 = self.encoder0_relu(self.encoder0_bn(self.encoder0(x)))
        e0_pool = self.maxpool(e0)
        
        e1 = self.encoder1(e0_pool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Ścieżka dekodowania (Decoder ze skip connection)
        d4 = self.upconv4(e4)
        d4 = self._pad_to_match(d4, e3)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = self._pad_to_match(d3, e2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = self._pad_to_match(d2, e1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = self._pad_to_match(d1, e0)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.decoder1(d1)
        
        d0 = self.upconv0(d1)
        d0 = self.decoder0(d0)
        
        out = self.final_conv(d0)
        # Zwracamy surowe logity. Aktywacja (Sigmoid) wędruje do funkcji straty lub post-processingu
        return out
