import torch
import urllib.request
from PIL import Image
from io import BytesIO
from unaiverse.agent import Agent
from unaiverse.dataprops import Data4Proc
from unaiverse.networking.node.node import Node


# Image offering network: a module that simpy downloads and offers an image as its output
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor | None = None):
        with urllib.request.urlopen("https://cataas.com/cat") as response:
            inp = Image.open(BytesIO(response.read()))
            inp.show()  # Let's see the pic (watch out: random pic with a cat somewhere)
            print(f"Downloaded image shape {inp.size}, type: {type(inp)}, expected-content: cat")
        return inp


# Agent
agent = Agent(proc=Net(),
              proc_inputs=[Data4Proc(data_type="all")],
              proc_outputs=[Data4Proc(data_type="img")],  # A PIL image is being "generated" here
              behav_lone_wolf="ask")

# Node hosting agent
node = Node(node_name="Test1", hosted=agent, hidden=True, clock_delta=1. / 30.)

# Telling this agent to connect to the ResNet one
node.ask_to_get_in_touch(node_name="Test0")

# Running node for 10 seconds
node.run(max_time=10.0)

# Printing the last received data from the ResNet agent
out = agent.get_last_streamed_data('Test0')[0]
print(f"Received response: {out}")  # Now we expect a textual response
print("")
print(f"Notice: instead of using this agent, you can also: search for the ResNet node (ResNetAgent) "
      f"in the UNaIVERSE portal, connect to it using our in-browser agent, select a picture from "
      f"your disk, send it to the agent, get back the text response!")
