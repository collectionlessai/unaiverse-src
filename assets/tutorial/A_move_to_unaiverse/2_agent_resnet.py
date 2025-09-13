import torch
import torchvision
from unaiverse.agent import Agent
from unaiverse.dataprops import Data4Proc
from unaiverse.networking.node.node import Node

# Downloading PyTorch module (ResNet)
net = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()

# Agent: we pass the network as "processor".
# Check the input and output properties of the processor, they are coherent with the
# input and output shapes of ResNet; here "None" means "whatever, but this axis must be
# there!". By default, this agent will act as a serving "lone wolf", serving whoever asks for
# a prediction.
agent = Agent(proc=net,
              proc_inputs=[Data4Proc(data_type="tensor", tensor_shape=(None, 3, None, None),
                                     tensor_dtype=torch.float32)],
              proc_outputs=[Data4Proc(data_type="tensor", tensor_shape=(None, 1000),
                                      tensor_dtype=torch.float32)])

# Node hosting agent: a node will be created in your account with this name, if not
# existing; it is "hidden" meaning that only you can see it in UNaIVERSE (since it is
# just a test!); the clock speed can be tuned accordingly to your needed and computing
# power.
node = Node(node_name="ResNetAgent", hosted=agent, hidden=True, clock_delta=1. / 30.)

# Running node (forever)
node.run()
