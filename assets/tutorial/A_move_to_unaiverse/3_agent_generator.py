import torch
from unaiverse.agent import Agent
from unaiverse.dataprops import Data4Proc
from unaiverse.networking.node.node import Node


# Custom generator network: a module that simply generates an image with
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
agent = Agent(proc=Net(),
              proc_inputs=[Data4Proc(data_type="all")],  # Able to get every type of data (since it won't use it :))
              proc_outputs=[Data4Proc(data_type="tensor", tensor_shape=(1, 3, 224, 224),
                                      tensor_dtype="torch.float32")],  # These are the properties of generator output
              )

# To retrieve the result we got from the ResNet agent, we define a hook
# that will be called at the end of every run cycle
def hook(_node: Node):
    # Printing the last received data from the ResNet agent
    _out = _node.agent.get_last_streamed_data('Test0')[0]
    if _out is not None:
        _node.agent.print(f"Received data shape: {tuple(_out.shape)}, dtype: {_out.dtype}")

# Node hosting agent
node = Node(node_name="Test1", hosted=agent, hidden=True, clock_delta=1. / 5., run_hook=hook)

# Running node for 10 seconds
node.run(get_in_touch="Test0", max_time=10.0)
