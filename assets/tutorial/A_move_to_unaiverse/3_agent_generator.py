import torch
from unaiverse.agent import Agent
from unaiverse.dataprops import Data4Proc
from unaiverse.networking.node.node import Node


# Custom generator network: a module that simpy generates an image with
# "random" pixel intensities; we will use this as processor of our new agent.
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # The input will be ignored, and a default None value is needed
    def forward(self, x: torch.Tensor | None = None):
        inp = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        print(f"Generated data shape: {tuple(inp.shape)}, dtype: {inp.dtype}")
        return inp


# Agent: we use the generator as processor.
# This agent will still be a "lone wolf", but we will force a different behavior, that is
# the one of "asking" another agent to handle the generated data, getting back a response.
agent = Agent(proc=Net(),
              proc_inputs=[Data4Proc(data_type="all")],  # Able to get every type of data (since it won't use it :))
              proc_outputs=[Data4Proc(data_type="tensor", tensor_shape=(1, 3, 224, 224),
                                      tensor_dtype="torch.float32")],  # These are the properties of generator output
              behav_lone_wolf="ask")  # Setting this behavior: "ask the partner to respond to your generated data"

# Node hosting agent
node = Node(node_name="GeneratorAgent", hosted=agent, hidden=True, clock_delta=1. / 30.)

# Telling this agent to connect to the ResNet one
node.ask_to_get_in_touch(node_name="ResNetAgent")

# Running node for 10 seconds
node.run(max_time=10.0)

# Printing the last received data from the ResNet agent
out = agent.get_last_streamed_data('ResNetAgent')[0]
print(f"Received data shape: {tuple(out.shape)}, dtype: {out.dtype}")
