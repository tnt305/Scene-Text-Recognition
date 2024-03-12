import torch
import torch.nn as nn
import timm
class CRNN(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        hidden_size, 
        n_layers, 
        dropout=0.2, 
        unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        # Khởi tạo pretrained model resnet101
        backbone = timm.create_model(
            'resnet101', 
            in_chans=1,
            pretrained=True
        )
        # Bỏ đi lớp classifer gốc của pretrained 
        modules = list(backbone.children())[:-2]
        # Thêm vào lớp AdaptiveAvgPool2d
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Mở băng (unfreeze) một số layers cuối cùng của pretrained model
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True
        
        # Layer dùng để map từ CNN features maps sang LSTM 
        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 1024),  
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
        self.lstm = nn.LSTM(
            1024, hidden_size, 
            n_layers, bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),  
            nn.LogSoftmax(dim=2)  
        )

    def forward(self, x):
        x = self.backbone(x) # shape: (bs, c, h, w)
        x = x.permute(0, 3, 1, 2) # shape: (bs, w, c, h)
        x = x.view(x.size(0), x.size(1), -1)  # Remove h: (bs, w, c)
        x = self.mapSeq(x) 
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.out(x) # shape: (bs, seq_len, n_classes)
        x = x.permute(1, 0, 2) # Based on CTC # (seq_len, bs, n_classes)
         
        return x