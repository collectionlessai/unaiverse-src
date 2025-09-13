import torchvision
import urllib.request
from unaiverse.agent import Agent
from unaiverse.dataprops import Data4Proc
from unaiverse.networking.node.node import Node

# Downloading PyTorch module (ResNet)
net = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()

# Getting transforms from PyTorch model
transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms

# Getting class names
with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
    c_names = [line.strip().decode('utf-8') for line in f.readlines()]

# Agent: we change the data type, to be able to handle stream of images (instead of tensors).
# We can customize the transformations from the streamed format to the processor inference
# format (every callable function is fine!). Similarly, we can customize the way we go from
# the actual output of the processor and what will be streamed (here we go from class
# probabilities to winning class name).
agent = Agent(proc=net,
              proc_inputs=[Data4Proc(data_type="img", stream_to_proc_transforms=transforms)],
              proc_outputs=[Data4Proc(data_type="text", proc_to_stream_transforms=lambda p: c_names[p.argmax(1)[0]])])

# Node hosting agent
node = Node(node_name="ResNetAgent", hosted=agent, hidden=True, clock_delta=1. / 30.)

# Running node
node.run()
