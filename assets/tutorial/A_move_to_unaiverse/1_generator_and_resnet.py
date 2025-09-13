import torch
import torchvision

# Downloading PyTorch module (ResNet)
net = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()

# Generating a random image (don't care about it, it is just a toy example,
# think it is a nice image!)
inp = torch.rand((1, 3, 224, 224), dtype=torch.float32)

# Inference: expects as input a tensor of type torch.float32, custom width and
# height, but 3 channels and batch dimension must be there; the output is a
# tensor with shape (1, 1000), i.e., a tensor in which batch dimension is
# present and then 1000 elements.
out = net(inp)

# Print shapes
print(f"Input shape: {tuple(inp.shape)}, dtype: {inp.dtype}")
print(f"Output shape: {tuple(out.shape)}, dtype: {out.dtype}")
