import torch

print(torch.cuda.is_available())      # 是否可用
print(torch.cuda.device_count())      # GPU 数量
print(torch.cuda.get_device_name(0))  # 第 1 张 GPU 名称