import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(img_size=224, num_classes=9, num_frames=8, attention_type='divided_space_time', pretrained=False)
x = torch.randn(4, 3, 8, 224, 224)  # (batch x channels x frames x height x width)
y = model(x)
print(y.shape)