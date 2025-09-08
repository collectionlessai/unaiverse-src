"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import os
import torch
import shutil
import numpy as np
import torchvision
from PIL import Image
import urllib.request
from typing import Callable
import torch.nn.functional as F
from unaiverse.dataprops import Data4Proc
from unaiverse.modules.cnu.cnus import CNUs
from unaiverse.modules.cnu.layers import LinearCNU
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from unaiverse.modules.utils import get_proc_inputs_and_proc_outputs_for_image_classification
from unaiverse.modules.utils import ModuleWrapper, transforms_factory, get_proc_inputs_and_proc_outputs_for_rnn


class RNNTokenLM(ModuleWrapper):

    def __init__(self, num_emb: int, emb_dim: int, y_dim: int, h_dim: int, batch_size: int = 1, seed: int = -1):
        super(RNNTokenLM, self).__init__(seed=seed)
        device = self.device
        u_dim = emb_dim
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

        self.proc_inputs = [
            Data4Proc(data_type="tensor", tensor_shape=(u_dim, ), tensor_dtype=torch.float32,
                      pubsub=False, private_only=True)
        ]
        self.proc_outputs = [
            Data4Proc(data_type="tensor", tensor_shape=(y_dim, ), tensor_dtype=torch.float32,
                      pubsub=False, private_only=True)
        ]

        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.B = torch.nn.Linear(u_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.h_init = torch.randn((batch_size, h_dim), device=device)
        self.u_init = torch.zeros((batch_size, u_dim), device=device)
        self.h = None
        self.y = None

    def forward(self, u: torch.Tensor | None = None, first: bool = True, last: bool = False):
        if first:
            h = self.h_init
            u = self.u_init if u is None else u
        else:
            h = self.h.detach()
            u = self.embeddings((torch.argmax(self.y.detach(), dim=1) if self.y.shape[1] > 1
                                 else self.y.squeeze(1).detach()).to(self.device))

        self.h = torch.tanh(self.A(h) + self.B(u))
        self.y = self.C(self.h)
        return self.y


class RNN(ModuleWrapper):

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, batch_size: int = 1, seed: int = -1):
        super(RNN, self).__init__(seed=seed)
        device = self.device
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.register_buffer('h_init', torch.randn((batch_size, h_dim), device=device))
        self.h = None
        self.u_dim = u_dim
        self.du_dim = du_dim

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False):
        if first:
            h = self.h_init.data
        else:
            h = self.h.detach()
        if u is None:
            u = torch.zeros((h.shape[0], self.u_dim), dtype=torch.float32, device=self.device)
        else:
            u = u.to(self.device)
        if du is None:
            du = torch.zeros((h.shape[0], self.du_dim), dtype=torch.float32, device=self.device)
        else:
            du = du.to(self.device)

        self.h = torch.tanh(self.A(h) + self.B(torch.cat([du, u], dim=1)))
        y = self.C(self.h)
        return y


class CSSM(ModuleWrapper):

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = F.tanh,
                 project_every: int = 0, local: bool = False, batch_size: int = 1, seed: int = -1):
        super(CSSM, self).__init__(seed=seed)
        device = self.device
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        self.batch_size = batch_size
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        # Define linear transformation matrices for state update and output mapping
        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)  # Recurrent weight matrix
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)  # Input-to-hidden mapping
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)  # Hidden-to-output mapping

        # Hidden state initialization
        self.register_buffer('h_init', torch.randn((batch_size, h_dim), device=device))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim), device=device))
        self.h = None
        self.dh = None
        self.sigma = sigma  # The non-linear activation function

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.delta = 1.  # Discrete time step
        self.local = local  # If True the state update is computed locally in time (i.e., kept out from the graph)
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """Placeholder for eigenvalue adjustment method."""
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False):
        """Forward pass that updates the hidden state and computes the output."""

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else (
            torch.zeros((self.batch_size, self.u_dim), device=self.device))
        du = du.to(self.device) if du is not None else (
            torch.zeros((self.batch_size, self.du_dim), device=self.device))

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data

        # Track the gradients on h from here on
        h.requires_grad_()

        # Check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # Handle inputs
        du, u = self.handle_inputs(du, u)

        # Update hidden state based on input and previous hidden state
        h_new = self.A(h) + self.B(torch.cat([du, u], dim=1))

        if self.local:

            # In the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:

            # In the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # Store the new state for the next iteration
        self.h_next.data = h_new.detach()
        self.forward_count += 1

        return y


class CDiagR(ModuleWrapper):
    """Diagonal matrix-based generator with real-valued transformations."""
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = lambda x: x,
                 project_every: int = 0, local: bool = False, batch_size: int = 1, seed: int = -1):
        super(CDiagR, self).__init__(seed=seed)
        device = self.device
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        self.batch_size = batch_size
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        # Define diagonal transformation and linear layers
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.float32)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # Hidden state initialization
        self.register_buffer('h_init', torch.randn((batch_size, h_dim), device=device))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim), device=device))
        self.h = None
        self.dh = None
        self.sigma = sigma  # The non-linear activation function

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.delta = 1.
        self.local = local  # If True the state update is computed locally in time (i.e., kept out from the graph)
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """Normalize the diagonal weight matrix by setting signs."""
        self.diag.weight.copy_(torch.sign(self.diag.weight))

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False):
        """Forward pass with diagonal transformation."""

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else (
            torch.zeros((self.batch_size, self.u_dim), device=self.device))
        du = du.to(self.device) if du is not None else (
            torch.zeros((self.batch_size, self.du_dim), device=self.device))

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data

        # Track the gradients on h from here on
        h.requires_grad_()

        # Check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # Handle inputs
        du, u = self.handle_inputs(du, u)

        # Apply diagonal transformation to hidden state
        h_new = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

        if self.local:

            # In the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:

            # In the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # Store the new state for the next iteration
        self.h_next.data = h_new.detach()
        self.forward_count += 1

        return y


class CDiagC(ModuleWrapper):
    """Diagonal matrix-based generator with complex-valued transformations."""
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = lambda x: x,
                 project_every: int = 0, local: bool = False, batch_size: int = 1, seed: int = -1):
        super(CDiagC, self).__init__(seed=seed)
        device = self.device
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        self.batch_size = batch_size
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        # Define diagonal transformation with complex numbers
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device, dtype=torch.cfloat)

        # Hidden state initialization
        self.register_buffer('h_init', torch.randn((batch_size, h_dim), device=device))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim), device=device))
        self.h = None
        self.dh = None
        self.sigma = sigma  # The non-linear activation function

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.delta = 1.
        self.local = local  # If True the state update is computed locally in time (i.e., kept out from the graph)
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """ Normalize the diagonal weight matrix by dividing by its magnitude. """
        self.diag.weight.div_(self.diag.weight.abs())

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False):
        """Forward pass with complex-valued transformation."""

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else torch.zeros((self.batch_size, self.u_dim),
                                                                           device=self.device, dtype=torch.cfloat)
        du = du.to(self.device) if du is not None else torch.zeros((self.batch_size, self.du_dim),
                                                                   device=self.device, dtype=torch.cfloat)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data

        # Track the gradients on h from here on
        h.requires_grad_()

        # Check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # Handle inputs
        du, u = self.handle_inputs(du, u)

        # Apply complex diagonal transformation
        h_new = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

        if self.local:

            # In the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:

            # In the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # Store the new state for the next iteration
        self.h_next.data = h_new.detach()
        self.forward_count += 1

        return y.real


class CTE(ModuleWrapper):
    """Antisymmetric Matrix Exponential Generator implementing continuous-time dynamics.

    Uses antisymmetric weight matrix with matrix exponential for stable hidden state evolution.

    Args:
        u_shape: Input shape (tuple of integers)
        d_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        local: Local computations (bool)
        seed: Random seed (positive int)
    """

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 cnu_memories: int = 0, batch_size: int = 1, seed: int = -1):
        super(CTE, self).__init__(seed=seed)
        device = self.device
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        self.batch_size = batch_size
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        # Antisymmetric weight matrix (W - W^T)
        self.W = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.Id = torch.eye(h_dim, device=device)  # Identity matrix

        # Input projection matrix
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)

        # Output projection matrix
        if cnu_memories <= 0:
            self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        else:
            self.C = LinearCNU(h_dim, y_dim, bias=False, device=device, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)

        # Hidden state initialization
        self.register_buffer('h_init', torch.randn((batch_size, h_dim), device=device))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim), device=device))
        self.h = None
        self.dh = None
        self.sigma = sigma  # The non-linear activation function

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.delta = delta
        self.local = local
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """Placeholder for eigenvalue adjustment method"""
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False) -> torch.Tensor:
        """Forward pass through the system dynamics.

        Args:
            u: Input tensor of shape (batch_size, u_dim)
            du: Input descriptor tensor of shape (batch_size, du_dim)
            first: Flag indicating first step (resets hidden state)
            last: Flag indicating last step (does nothing)

        Returns:
            y: Output tensor of shape (batch_size, y_dim)
        """

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else (
            torch.zeros((self.batch_size, self.u_dim), device=self.device))
        du = du.to(self.device) if du is not None else (
            torch.zeros((self.batch_size, self.du_dim), device=self.device))

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data

        # Track the gradients on h from here on
        h.requires_grad_()

        # Check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        if not isinstance(self.C, LinearCNU):
            C = self.C
        else:
            udu = torch.cat([du, u], dim=1)
            weight_C = self.C.compute_weights(udu).view(self.C.out_features, self.C.in_features)

            def C(x):
                return torch.nn.functional.linear(x, weight_C)

        # Handle inputs
        du, u = self.handle_inputs(du, u)

        # Antisymmetric matrix construction
        A = 0.5 * (self.W.weight - self.W.weight.t())
        A_expm = torch.linalg.matrix_exp(A * self.delta)  # Matrix exponential
        rec = F.linear(h, A_expm, self.W.bias)  # Recurrent component

        # Input processing component
        A_inv = torch.linalg.inv(A)
        inp = A_inv @ (A_expm - self.Id) @ self.B(torch.cat([du, u], dim=1)).unsqueeze(-1)

        # Handle locality
        h_new = rec + inp.squeeze(-1)  # Updated hidden state
        if self.local:

            # In the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:

            # In the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta

        # Compute output using a nonlinear activation function
        y = C(self.sigma(self.h))

        # Store the new state for the next iteration
        self.h_next.data = h_new
        self.forward_count += 1

        return y


class CTEInitStateBZeroInput(CTE):

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 cnu_memories: int = 0, batch_size: int = 1, seed: int = -1):
        super(CTEInitStateBZeroInput, self).__init__(u_shape, d_dim, y_dim, h_dim, delta, sigma, project_every,
                                                     local, cnu_memories, batch_size, seed)

    @torch.no_grad()
    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.B(udu).detach() / torch.sum(udu, dim=1)

    @staticmethod
    def handle_inputs(du, u):
        return torch.zeros_like(du), torch.zeros_like(u)


class CTEToken(CTE):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int, y_dim: int, h_dim: int, seed: int = -1):
        super(CTEToken, self).__init__((emb_dim,), d_dim, y_dim, h_dim, delta=1.0, local=False, seed=seed)
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False):
        if u is not None:
            u = self.embeddings(u.to(self.device))
        y = super().forward(u, du, first=first, last=last)
        return y


class CTB(ModuleWrapper):
    """Block Antisymmetric Generator using 2x2 parameterized rotation blocks.

    Implements structured antisymmetric dynamics through learnable rotational frequencies.

    Args:
        u_shape: Input shape (tuple of integers)
        d_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        alpha: Dissipation added on the diagonal (also controls the eigenvalue projections method)
    """

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float = None,
                 alpha: float = 0., sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 batch_size: int = 1, seed: int = -1):
        super(CTB, self).__init__(seed=seed)
        device = self.device
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        self.batch_size = batch_size
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"
        self.order = h_dim // 2  # Number of 2x2 blocks

        # Learnable rotational frequencies
        self.omega = torch.nn.Parameter(torch.empty(self.order, device=device))
        self.register_buffer('ones', torch.ones(self.order, requires_grad=False, device=device))

        # Projection matrices
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # Damping configuration
        if alpha > 0.:

            # In this case we want to add the feedback parameter alpha and use it to move eigenvalues on the unit circle
            self.project_method = 'const'
            self.register_buffer('alpha', torch.full_like(self.omega.data, alpha, device=device))
        elif alpha == 0.:

            # This is the case in which we want to divide by the modulus
            self.project_method = 'modulus'
            self.register_buffer('alpha', torch.zeros_like(self.omega.data, device=device))
        elif alpha == -1.:
            self.project_method = 'alpha'
            self.register_buffer('alpha', torch.zeros_like(self.omega.data, device=device))

        # Hidden state initialization
        self.register_buffer('h_init', torch.randn((batch_size, h_dim), device=device))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim), device=device))
        self.h = None
        self.dh = None
        self.sigma = sigma  # The non-linear activation function

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.delta = delta
        self.local = local  # If True the state update is computed locally in time (i.e., kept out from the graph)
        self.reset_parameters()
        self.forward_count = 0
        self.project_every = project_every

    def reset_parameters(self) -> None:
        """Initialize rotational frequencies with uniform distribution"""
        torch.nn.init.uniform_(self.omega)

    @torch.no_grad()
    def adjust_eigs(self):
        """Adjust eigenvalues to maintain stability"""
        with torch.no_grad():
            if self.project_method == 'alpha':

                # Compute damping to maintain eigenvalues on unit circle
                self.alpha.copy_((1. - torch.sqrt(1. - (self.delta * self.omega) ** 2) / self.delta))
            elif self.project_method == 'modulus':

                # Normalize by modulus for unit circle stability
                module = torch.sqrt(self.ones ** 2 + (self.delta * self.omega) ** 2)
                self.omega.div_(module)
                self.ones.div_(module)

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False) -> torch.Tensor:
        """Forward pass through block-structured dynamics"""

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None \
            else torch.zeros((self.batch_size, self.u_dim), device=self.device)
        du = du.to(self.device) if du is not None \
            else torch.zeros((self.batch_size, self.du_dim), device=self.device)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data

        # Track the gradients on h from here on
        h.requires_grad_()
        h_pair = h.view(-1, self.order, 2)  # Reshape to (batch, blocks, 2)

        # Check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # Handle inputs
        du, u = self.handle_inputs(du, u)

        # Block-wise rotation with damping
        h1 = (self.ones - self.delta * self.alpha) * h_pair[..., 0] + self.delta * self.omega * h_pair[..., 1]
        h2 = -self.delta * self.omega * h_pair[..., 0] + (self.ones - self.delta * self.alpha) * h_pair[..., 1]

        # Recurrent and input components
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)
        inp = self.delta * self.B(torch.cat([du, u], dim=1))

        # Handle locality
        h_new = rec + inp  # Updated hidden state
        if self.local:

            # In the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:

            # In the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # Store the new state for the next iteration
        self.h_next.data = h_new.detach()
        self.forward_count += 1

        return y


class CTBE(ModuleWrapper):
    """Antisymmetric Generator with Exact Matrix Exponential Blocks.

    Implements precise rotational dynamics using trigonometric parameterization.

    Args:
        u_shape: Input shape (tuple of integers)
        d_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
    """

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 cnu_memories: int = 0, batch_size: int = 1, seed: int = -1):
        super(CTBE, self).__init__(seed=seed)
        device = self.device
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        self.batch_size = batch_size
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"
        self.order = h_dim // 2

        # Learnable rotational frequencies
        self.omega = torch.nn.Parameter(torch.empty(self.order, device=device))
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        if cnu_memories <= 0:
            self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        else:
            self.C = LinearCNU(h_dim, y_dim, bias=False, device=device, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)

        # Hidden state initialization
        self.register_buffer('h_init', torch.randn((batch_size, h_dim), device=device))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim), device=device))
        self.h = None
        self.dh = None
        self.sigma = sigma  # The non-linear activation function

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.delta = delta
        self.local = local  # If True the state update is computed locally in time (i.e., kept out from the graph)
        self.reset_parameters()
        self.forward_count = 0
        self.project_every = project_every

    def reset_parameters(self) -> None:
        """Initialize rotational frequencies"""
        if not isinstance(self.omega, CNUs):
            torch.nn.init.uniform_(self.omega)
        else:
            torch.nn.init.uniform_(self.omega.M)

    @torch.no_grad()
    def adjust_eigs(self):
        """Placeholder for eigenvalue adjustment"""
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True, last: bool = False) -> torch.Tensor:
        """Exact matrix exponential forward pass"""

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None \
            else torch.zeros((self.batch_size, self.u_dim), device=self.device)
        du = du.to(self.device) if du is not None \
            else torch.zeros((self.batch_size, self.du_dim), device=self.device)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data

        # Track the gradients on h from here on
        h.requires_grad_()
        h_pair = h.view(-1, self.order, 2)

        # Check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        if not isinstance(self.C, LinearCNU):
            C = self.C
        else:
            udu = torch.cat([du, u], dim=1)
            weight_C = self.C.compute_weights(udu).view(self.C.out_features, self.C.in_features)

            def C(x):
                return torch.nn.functional.linear(x, weight_C)

        # Handle inputs
        du, u = self.handle_inputs(du, u)
        udu = torch.cat([du, u], dim=1)

        # Trigonometric terms for exact rotation
        cos_t = torch.cos(self.omega * self.delta)
        sin_t = torch.sin(self.omega * self.delta)

        # Rotational update
        h1 = cos_t * h_pair[..., 0] + sin_t * h_pair[..., 1]
        h2 = -sin_t * h_pair[..., 0] + cos_t * h_pair[..., 1]
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)

        # Input processing
        u_hat = self.B(udu).view(-1, self.order, 2)
        inp1 = (sin_t * u_hat[..., 0] - (cos_t - 1) * u_hat[..., 1]) / self.omega
        inp2 = ((cos_t - 1) * u_hat[..., 0] + sin_t * u_hat[..., 1]) / self.omega
        inp = torch.stack([inp1, inp2], dim=-1).flatten(start_dim=1)

        # Handle locality
        h_new = rec + inp  # Updated hidden state
        if self.local:

            # In the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:

            # In the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta

        # Compute output using a nonlinear activation function
        y = C(self.sigma(self.h))

        # Store the new state for the next iteration
        self.h_next.data = h_new.detach()
        self.forward_count += 1

        return y


class CTBEInitStateBZeroInput(CTBE):
    def __init__(self, u_shape, d_dim, y_dim, h_dim, delta, local, cnu_memories: int = 0,
                 batch_size: int = 1, seed: int = -1):
        super().__init__(u_shape=u_shape, d_dim=d_dim, y_dim=y_dim, h_dim=h_dim, delta=delta, local=local,
                         cnu_memories=cnu_memories, batch_size=batch_size, seed=seed)

    @torch.no_grad()
    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.B(udu).detach() / torch.sum(udu)

    @staticmethod
    def handle_inputs(du, u):
        return torch.zeros_like(du), torch.zeros_like(u)


class CNN(ModuleWrapper):

    def __init__(self, d_dim: int, in_channels: int = 3, in_res: int = 32, return_input: bool = False,
                 seed: int = -1):
        super(CNN, self).__init__(seed=seed)
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        self.transforms = transforms_factory("rgb" + str(in_res) if in_channels == 3 else "gray" + str(in_res))

        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, d_dim),
            torch.nn.Sigmoid()
        ).to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if not self.return_input:
            return o
        else:
            return y, o


class CNNCNU(ModuleWrapper):

    def __init__(self, d_dim: int, cnu_memories: int, in_channels: int = 3, in_res: int = 32,
                 delta: int = 1, scramble: bool = False, return_input: bool = False, seed: int = -1):
        super(CNNCNU, self).__init__(seed=seed)
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        self.transforms = transforms_factory("rgb" + str(in_res) if in_channels == 3 else "gray" + str(in_res))

        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(2048),
            torch.nn.ReLU(inplace=True),
            LinearCNU(2048, d_dim, key_mem_units=cnu_memories, delta=delta, scramble=scramble),
            torch.nn.Sigmoid()
        ).to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if not self.return_input:
            return o
        else:
            return y, o


class SingleLayerCNU(ModuleWrapper):

    def __init__(self, d_dim: int, cnu_memories: int, in_channels: int = 3, in_res: int = 32,
                 delta: int = 1, scramble: bool = False, return_input: bool = False, seed: int = -1):
        super(SingleLayerCNU, self).__init__(seed=seed)
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        self.transforms = transforms_factory("rgb" + str(in_res) if in_channels == 3 else "gray" + str(in_res))

        self.module = torch.nn.Sequential(
            torch.nn.Flatten(),
            LinearCNU(in_res * in_res * in_channels, d_dim, key_mem_units=cnu_memories, delta=delta, scramble=scramble),
            torch.nn.Sigmoid()
        ).to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if not self.return_input:
            return o
        else:
            return y, o


class CNNMNIST(CNN):

    def __init__(self, *args, **kwargs):
        kwargs['in_channels'] = 1
        kwargs['in_res'] = 28
        super(CNNMNIST, self).__init__(*args, **kwargs)
        self.transforms = transforms_factory("gray_mnist")


class CNNCNUMNIST(CNNCNU):

    def __init__(self, *args, **kwargs):
        kwargs['in_channels'] = 1
        kwargs['in_res'] = 28
        super(CNNCNUMNIST, self).__init__(*args, **kwargs)
        self.transforms = transforms_factory("gray_mnist")


class SingleLayerCNUMNIST(SingleLayerCNU):

    def __init__(self, *args, **kwargs):
        kwargs['in_channels'] = 1
        kwargs['in_res'] = 28
        super(SingleLayerCNUMNIST, self).__init__(*args, **kwargs)
        self.transforms = transforms_factory("gray_mnist")


class ResNet(ModuleWrapper):
    def __init__(self, d_dim: int = -1, return_input: bool = False, seed: int = -1, freeze_backbone: bool = True):
        super(ResNet, self).__init__(seed=seed)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        self.transforms = transforms_factory("rgb224")
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for layer in resnet.parameters():
                if layer != resnet.fc:
                    layer.requires_grad = False

        if d_dim > 0:
            resnet.fc = torch.nn.Sequential(
                torch.nn.Linear(resnet.fc.in_features, d_dim),
                torch.nn.Sigmoid()
            )

        self.module = resnet.to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if not self.return_input:
            return o
        else:
            return y, o


class ResNetCNU(ModuleWrapper):
    def __init__(self, d_dim: int, cnu_memories: int,
                 delta: int = 1, scramble: bool = False, return_input: bool = False, seed: int = -1):
        super(ResNetCNU, self).__init__(seed=seed)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        self.transforms = transforms_factory("rgb224")
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")

        resnet.fc = torch.nn.Sequential(
            LinearCNU(resnet.fc.in_features, d_dim, key_mem_units=cnu_memories, delta=delta, scramble=scramble),
            torch.nn.Sigmoid()
        )

        self.module = resnet.to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if not self.return_input:
            return o
        else:
            return y, o


class ViT(ModuleWrapper):
    def __init__(self, d_dim: int = -1, return_input: bool = False, seed: int = -1):
        super(ViT, self).__init__(seed=seed)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        self.transforms = torchvision.transforms.Compose([
            weights.transforms(),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
        ])
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        vit = torchvision.models.vit_b_16(weights=weights)

        if d_dim > 0:
            vit.heads = torch.nn.Sequential(
                torch.nn.Linear(vit.heads.head.in_features, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, d_dim),
                torch.nn.Sigmoid()
            )
            self.labels = ["unk"] * d_dim
        else:
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            self.labels = []
            with urllib.request.urlopen(url) as f:
                self.labels = [line.strip().decode('utf-8') for line in f.readlines()]

        self.module = vit.to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if not self.return_input:
            return o
        else:
            return y, o


class DenseNet(ModuleWrapper):
    def __init__(self, d_dim: int = -1, return_input: bool = False, seed: int = -1):
        super(DenseNet, self).__init__(seed=seed)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        self.transforms = transforms_factory("rgb224")
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        densenet = torchvision.models.densenet121(weights=None)

        if d_dim > 0:
            densenet.classifier = torch.nn.Sequential(
                torch.nn.Linear(densenet.classifier.in_features, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, d_dim),
                torch.nn.Sigmoid()
            )

        self.module = densenet.to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if not self.return_input:
            return o
        else:
            return y, o


class EfficientNet(ModuleWrapper):
    def __init__(self, d_dim: int = -1, return_input: bool = False, seed: int = -1):
        super(EfficientNet, self).__init__(seed=seed)
        self.return_input = return_input
        if self.return_input:
            self.proc_outputs.insert(0, Data4Proc(data_type="img", pubsub=False, private_only=True))
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.transforms = weights.transforms
        self.proc_inputs, self.proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        effnet = torchvision.models.efficientnet_b0(weights=weights)

        if d_dim > 0:
            effnet.classifier = torch.nn.Sequential(
                torch.nn.Linear(effnet.classifier[1].in_features, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, d_dim),
                torch.nn.Sigmoid()
            )

        self.module = effnet.to(self.device)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module(self.transforms(y).to(self.device))
        if o.dim() == 1:
            o = o.unsqueeze(0)
        if not self.return_input:
            return o
        else:
            return y, o


class FasterRCNN(ModuleWrapper):
    def __init__(self, seed: int = -1):
        self.labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                       'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                       'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                       'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                       'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                       'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                       ]

        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        faster_rcnn.eval()
        self.transforms = torchvision.transforms.Compose([transforms_factory("rgb-no_norm"),
                                                          torchvision.transforms.Lambda(lambda x: x.squeeze(0)),
                                                          weights.transforms()])

        super(FasterRCNN, self).__init__(
            module=faster_rcnn,
            proc_inputs=[Data4Proc(data_type="img", pubsub=False, private_only=True)],
            proc_outputs=[Data4Proc(data_type="tensor", tensor_dtype=torch.long, tensor_shape=(None,),
                                    pubsub=False, private_only=True),
                          Data4Proc(data_type="tensor", tensor_dtype=torch.float32, tensor_shape=(None,),
                                    pubsub=False, private_only=True),
                          Data4Proc(data_type="tensor", tensor_dtype=torch.float32, tensor_shape=(None, 4),
                                    pubsub=False, private_only=True),
                          Data4Proc(data_type="text",
                                    pubsub=False, private_only=True)],
            seed=seed)

    def forward(self, y: Image.Image, first: bool = True, last: bool = False):
        o = self.module([self.transforms(y).to(self.device)])  # List with 1 image per element (no batch dim)

        found_class_indices = o[0]['labels']
        found_class_scores = o[0]['scores']
        found_class_boxes = o[0]['boxes']
        valid = found_class_scores > 0.8

        found_class_indices = found_class_indices[valid]
        found_class_scores = found_class_scores[valid]
        found_class_boxes = found_class_boxes[valid]
        found_class_names = [self.labels[i.item()] for i in found_class_indices]

        return found_class_indices, found_class_scores, found_class_boxes, ", ".join(found_class_names)


class TinyLLama(ModuleWrapper):
    def __init__(self):
        super(TinyLLama, self).__init__(
            proc_inputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)],
            proc_outputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)]
        )
        self.module = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                               torch_dtype=torch.bfloat16, device=self.device)

    def forward(self, msg: str, first: bool = False, last: bool = False):
        msg_struct = [{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content": msg}]
        prompt = self.module.tokenizer.apply_chat_template(msg_struct, tokenize=False, add_generation_prompt=True)

        out = self.module(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        out = out[0]["generated_text"] if (out is not None and len(out) > 0 and "generated_text" in out[0])\
            else "Error!"
        if "<|assistant|>\n" in out:
            out = out.split("<|assistant|>\n")[1]
        return out.strip()


class LLama(ModuleWrapper):
    def __init__(self):
        super(LLama, self).__init__(
            proc_inputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)],
            proc_outputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)]
        )
        self.module = pipeline("text-generation", model="meta-llama/Llama-3.2-3B",
                               torch_dtype=torch.bfloat16, device=self.device)

    def forward(self, msg: str, first: bool = False, last: bool = False):
        msg_struct = [{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content": msg}]
        prompt = self.module.tokenizer.apply_chat_template(msg_struct, tokenize=False, add_generation_prompt=True)

        out = self.module(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        out = out[0]["generated_text"] if (out is not None and len(out) > 0 and "generated_text" in out[0])\
            else "Error!"
        if "<|assistant|>\n" in out:
            out = out.split("<|assistant|>\n")[1]
        return out.strip()


class Phi(ModuleWrapper):
    def __init__(self):
        super(Phi, self).__init__(
            proc_inputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)],
            proc_outputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)]
        )
        self.module = pipeline("text-generation", model="microsoft/Phi-3.5-mini-instruct",
                               torch_dtype="auto", device=self.device)

    def forward(self, msg: str, first: bool = False, last: bool = False):
        msg_struct = [{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content": msg}]
        prompt = self.module.tokenizer.apply_chat_template(msg_struct, tokenize=False, add_generation_prompt=True)

        out = self.module(prompt, max_new_tokens=256, do_sample=True, return_full_text=False)
        out = out[0]["generated_text"] if (out is not None and len(out) > 0 and "generated_text" in out[0])\
            else "Error!"
        if "<|assistant|>\n" in out:
            out = out.split("<|assistant|>\n")[1]
        return out.strip()


class LangSegmentAnything(ModuleWrapper):
    def __init__(self):
        super(LangSegmentAnything, self).__init__(
            proc_inputs=[Data4Proc(data_type="img", pubsub=False, private_only=True),
                         Data4Proc(data_type="text", pubsub=False, private_only=True)],
            proc_outputs=[Data4Proc(data_type="img", pubsub=False, private_only=True)]
        )
        from lang_sam import LangSAM
        self.module = LangSAM(device=self.device)

        # Generate a 64x64 error image (with text "Error" on it)
        from PIL import ImageDraw, ImageFont
        self.error_img = Image.new("RGB", (64, 64), color="white")
        draw = ImageDraw.Draw(self.error_img)
        font = ImageFont.load_default()
        text = "Error"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((64 - text_width) // 2, (64 - text_height) // 2)
        draw.text(position, text, fill="black", font=font)

    def forward(self, image_pil: Image, msg: str, first: bool = False, last: bool = False):
        try:
            image_pil = image_pil.convert("RGB") if image_pil.mode != "RGB" else image_pil  # Forcing RGB
            out = self.module.predict([image_pil], [msg])

            if (out is None or not isinstance(out, list) or len(out) < 1 or
                    not isinstance(out[0], dict) or 'masks' not in out[0]) or out[0]['masks'].ndim != 3:
                return image_pil
            else:
                return LangSegmentAnything.__highlight_masks_on_image(image_pil, out[0]['masks'])
        except Exception as e:
            return self.error_img

    @staticmethod
    def __highlight_masks_on_image(image_pil: Image.Image, masks: np.ndarray, alpha: float = 0.75):
        img_np = np.array(image_pil, dtype=np.float32) / 255.0
        height, width, _ = img_np.shape
        num_masks = masks.shape[0]

        overlay_np = np.zeros((height, width, 3), dtype=np.float32)
        alpha_mask_combined = np.zeros((height, width, 1), dtype=np.float32)

        color_palette = [
            (255, 102, 102),  # Light Red
            (102, 255, 102),  # Light Green
            (102, 102, 255),  # Light Blue
            (255, 255, 102),  # Light Yellow
            (255, 102, 255),  # Light Magenta
            (102, 255, 255),  # Light Cyan
            (255, 178, 102),  # Orange
            (178, 102, 255),  # Purple
            (102, 178, 255),  # Sky Blue
        ]

        for i in range(num_masks):
            mask = masks[i, :, :].astype(np.bool)

            color_rgb_int = color_palette[i % len(color_palette)]
            color = np.array(color_rgb_int, dtype=np.float32) / 255.0
            overlay_np[mask] = (1 - alpha) * overlay_np[mask] + alpha * color
            alpha_mask_combined[mask] = np.maximum(alpha_mask_combined[mask], alpha)

        # Final blending and conversion ...
        final_np = (1 - alpha_mask_combined) * img_np + alpha_mask_combined * overlay_np
        final_np = (final_np * 255).astype(np.uint8)
        final_image = Image.fromarray(final_np)
        return final_image


class SmolVLM(ModuleWrapper):
    def __init__(self):
        super(SmolVLM, self).__init__(
            proc_inputs=[Data4Proc(data_type="img", pubsub=False, private_only=True),
                         Data4Proc(data_type="text", pubsub=False, private_only=True)],
            proc_outputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)]
        )
        model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        self.pre_post_processor = AutoProcessor.from_pretrained(model_id, device_map=self.device)

        from transformers import AutoModelForImageTextToText
        self.module = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                                  device_map=self.device)
        self.module = self.module.to(self.device)

    def forward(self, image_pil: Image, msg: str = "what is this?", first: bool = False, last: bool = False):
        image_pil = image_pil.convert("RGB") if image_pil.mode != "RGB" else image_pil  # Forcing RGB

        msg_struct = [{"role": "user", "content": [{"type": "text", "text": f"{msg}"},
                                                   {"type": "image", "image": image_pil}]}]

        prompt = self.pre_post_processor.apply_chat_template(msg_struct,
                                                             tokenize=True,
                                                             add_generation_prompt=True,
                                                             return_dict=True,
                                                             return_tensors="pt").to(self.device, dtype=torch.bfloat16)

        out = self.module.generate(**prompt, do_sample=False, max_new_tokens=128)
        out = self.pre_post_processor.batch_decode(out, skip_special_tokens=True)[0] if out is not None else "Error!"
        if "Assistant:" in out:
            out = out.split("Assistant:")[1]
        return out.strip()


class SiteRAG(ModuleWrapper):

    def __init__(self,
                 site_url: str,
                 site_folder: str = os.path.join("rag", "downloaded_site"),
                 db_folder: str = os.path.join("rag", "chroma_db")):
        super(SiteRAG, self).__init__(
            proc_inputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)],
            proc_outputs=[Data4Proc(data_type="text", pubsub=False, private_only=True)],
        )

        # Saving options
        self.site_url = site_url
        self.site_folder = site_folder
        self.db_folder = db_folder

        # Loading neural model
        model_id = "TheBloke/vicuna-7b-1.1-HF"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                                     device_map=self.device, offload_folder="offload")
        self.module = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

        # Embedder
        from langchain.embeddings import SentenceTransformerEmbeddings
        self.embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2",
                                                      model_kwargs={"device": self.device.type})

        # Crawling site
        self.crawl_website()
        self.crawled_site_to_rag_knowledge_base()

        # Setting up RAG stuff
        from langchain.vectorstores import Chroma
        db = Chroma(persist_directory=db_folder, embedding_function=self.embedder)
        self.retriever = db.as_retriever(search_kwargs={"k": 3})

    def forward(self, msg: str, first: bool = False, last: bool = False):

        # Build context
        docs = self.retriever.get_relevant_documents(msg)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {msg}\nAnswer:"

        # Generate answer
        out = self.module(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        out = out[0]['generated_text'][len(prompt):].strip() if (out is not None and len(out) > 0 and
                                                                 "generated_text" in out[0]) else "Error!"

        # Append source URLs
        best_doc_with_score = self.retriever.vectorstore.similarity_search_with_score(msg, k=1)
        best_doc, _ = best_doc_with_score[0]
        docs = [best_doc]
        sources = set("<a href='" +
                      doc.metadata['source'] +
                      "' onclick='window.open(this.href); return false;' style='color: blue;'>" +
                      doc.metadata['source'] + "</a>" for doc in docs)
        sources_text = "<br/><br/>\nURLs:\n" + "\n".join(sources)

        return out.strip() + sources_text

    def crawl_website(self, max_pages=300):
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        if os.path.exists(self.site_folder):
            shutil.rmtree(self.site_folder)
        os.makedirs(self.site_folder)
        visited = set()
        to_visit = [self.site_url]

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                r = requests.get(url, timeout=10)
                if "text/html" not in r.headers.get("Content-Type", ""):
                    continue

                parsed = urlparse(url)
                filename = parsed.path.strip("/") or "index.html"
                filename += ".crawled"
                file_path = os.path.join(self.site_folder, filename.replace("/", "__"))
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(r.text)

                soup = BeautifulSoup(r.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(url, link["href"])
                    if full_url.startswith(self.site_url) and full_url not in visited:
                        to_visit.append(full_url)
            except Exception as e:
                print(f"Error fetching {url}: {e}")

        print(f"Crawled {len(visited)} pages.")

    def crawled_site_to_rag_knowledge_base(self):
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
        from langchain.vectorstores import Chroma
        from langchain.docstore.document import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        docs = []
        for filename in os.listdir(self.site_folder):
            if filename.endswith(".crawled"):
                file_path = os.path.join(self.site_folder, filename)
                with open(file_path, encoding="utf-8") as f:
                    html = f.read()

                soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)  # Type: ignore

                page_path = filename.replace("__", "/").replace(".crawled", "")
                url = urljoin(self.site_url, page_path)

                docs.append(Document(page_content=text, metadata={"source": url}))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        chroma_db = Chroma.from_documents(split_docs, self.embedder, persist_directory=self.db_folder)
        chroma_db.persist()
