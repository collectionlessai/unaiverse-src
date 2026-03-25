import torch
import urllib.request
from PIL import Image
from io import BytesIO
from unaiverse.agent import Agent
from unaiverse.streams.dataprops import StreamType
from unaiverse.networking.node.node import Node


# Image offering network: a module that simpy downloads and offers an image as its output
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor | None = None):
        with urllib.request.urlopen("https://cataas.com/cat") as response:
            inp = Image.open(BytesIO(response.read()))
            # inp.show()  # Let's see the pic (watch out: random pic with a cat somewhere)
            print(f"Downloaded image shape {inp.size}, type: {type(inp)}, expected-content: cat")
        return inp


# Agent
agent = Agent(proc=Net(),
              proc_inputs=[StreamType(data_type="all")],
              proc_outputs=[StreamType(data_type="img")],  # A PIL image is being "generated" here
              behav_lone_wolf="ask")

# To retrieve the result we got from the ResNet agent, we define a hook
# that will be called at the end of every run cycle
def hook(_node: Node):
    # Printing the last received data from the ResNet agent
    out = _node.agent.get_last_streamed_data('Test0')[0]
    _node.agent.print(f"Received response: {out}")  # Now we expect a textual response
    _node.agent.print("")
    _node.agent.print(f"Notice: instead of using this agent, you can also: search for the ResNet node (ResNetAgent) "
                      f"in the UNaIVERSE portal, connect to it using our in-browser agent, select a picture from "
                      f"your disk, send it to the agent, get back the text response!")

# Node hosting agent
node = Node(node_name="Test1", hosted=agent, hidden=True, clock_delta=1. / 5., run_hook=hook)

# Running node for 45 seconds
node.run(max_time=45.0, get_in_touch="Test0")
