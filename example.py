import torch
from timesformer.models.vit import TimeSformer, MyTimeSformer, MyTimeSformerV2

def test_timesformer():
    model = TimeSformer(img_size=192, num_classes=9, num_frames=30, attention_type='divided_space_time',
                        pretrained=False)
    x = torch.randn(1, 3, 30, 192, 192)  # (batch x channels x frames x height x width)
    y = model(x)
    print(y.shape)

def test_my_timesformer():
    model = MyTimeSformer(img_size=192, num_classes=9, num_frames=3000, attention_type='divided_space_time')
    # [batch_size, time, transmitter, receiver, subcarrier]
    x = torch.randn(1, 3000, 3, 3, 30)
    y, y1 = model(x)
    print(f"Output dtype: {y.dtype}")
    print(f"Output dtype: {y1.dtype}")
    return y, y1

def test_my_timesformer_v2():
    model = MyTimeSformerV2(img_size=96, num_classes=10, num_frames=3000, attention_type='divided_space_time')
    # [batch_size, time, transmitter, receiver, subcarrier]
    x = torch.randn(1, 3000, 3, 3, 30)
    y, y1 = model(x)
    print(f"Output dtype: {y.dtype}")
    print(f"Output dtype: {y1.dtype}")
    return y, y1

if __name__ == "__main__":
    y, y1 = test_my_timesformer_v2()
    print(y.shape, y1.shape)
    print(y)
    print(y1)
