import torch
config={'backbone':'ViT-B/32',
    'activation':'relu',
    'batch_size':4,
    'num_classes':9,
    'device':'cuda'
}
mean1 = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=config['device'])
mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=config['device']).reshape(1, 3, 1, 1)
print(mean1.shape,mean.shape)
print(mean1)
print('#########')
print(mean)