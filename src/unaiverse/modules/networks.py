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
import sys
import time
import json
import torch
import shutil
import socket
import numpy as np
import torchvision
from PIL import Image
import urllib.request
from typing import Callable
import torch.nn.functional as F
from unaiverse.utils.logger import log
from unaiverse.modules.cnu.cnus import CNUs
from unaiverse.modules.cnu.layers import LinearCNU
from unaiverse.streams.dataprops import StreamType
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from unaiverse.modules.utils import get_proc_inputs_and_proc_outputs_for_image_classification, APIGatewayServer
from unaiverse.modules.utils import ModuleWrapper, transforms_factory, get_proc_inputs_and_proc_outputs_for_rnn


class RNNTokenLM(ModuleWrapper):
    def __init__(self, num_emb: int, emb_dim: int, y_dim: int, h_dim: int, batch_size: int = 1, *args, **kwargs):

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(num_emb, emb_dim)
                self.A = torch.nn.Linear(h_dim, h_dim, bias=False)
                self.B = torch.nn.Linear(emb_dim, h_dim, bias=False)
                self.C = torch.nn.Linear(h_dim, y_dim, bias=False)
                self.h_init = torch.randn((batch_size, h_dim))
                self.u_init = torch.zeros((batch_size, emb_dim))
                self.h = None
                self.y = None

            def forward(self, u: torch.Tensor | None = None, first: bool = True):
                if first:
                    h = self.h_init
                    u = self.u_init if (u is None or not isinstance(u, torch.Tensor) or u.shape != self.u_init.shape) \
                        else u
                else:
                    h = self.h.detach()
                    y_pred = torch.argmax(self.y.detach(), dim=-1).view(-1) if self.y.shape[-1] > 1 else self.y.detach()
                    u = self.embeddings(y_pred)

                self.h = torch.tanh(self.A(h) + self.B(u))
                self.y = self.C(self.h)
                return self.y

        super(RNNTokenLM, self).__init__(module=Net(),
                                         proc_inputs=[StreamType(data_type="tensor", tensor_shape=(1,),
                                                                 tensor_dtype=torch.long,
                                                                 pubsub=False, private_only=False)],
                                         proc_outputs=[StreamType(data_type="tensor", tensor_shape=(y_dim,),
                                                                  tensor_dtype=torch.float32,
                                                                  pubsub=False, private_only=False)],
                                         *args, **kwargs)


class RNN(ModuleWrapper):
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, batch_size: int = 1, device=None,
                 *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        class Net(torch.nn.Module):
            def __init__(self, _device):
                super().__init__()
                self.A = torch.nn.Linear(h_dim, h_dim, bias=False)
                self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False)
                self.C = torch.nn.Linear(h_dim, y_dim, bias=False)
                self.register_buffer('h_init', torch.randn((batch_size, h_dim)))
                self.h = None
                self.u_dim = u_dim
                self.du_dim = du_dim
                self._device = _device

            def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True):
                if first:
                    h = self.h_init.data
                else:
                    h = self.h.detach()
                if u is None:
                    u = torch.zeros((h.shape[0], self.u_dim), dtype=torch.float32, device=self._device)
                else:
                    u = u.to(self._device)
                if du is None:
                    du = torch.zeros((h.shape[0], self.du_dim), dtype=torch.float32, device=self._device)
                else:
                    du = du.to(self._device)

                self.h = torch.tanh(self.A(h) + self.B(torch.cat([du, u], dim=1)))
                y = self.C(self.h)
                return y

        # Populate self.device
        self.guess_device(device)

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)
        super(RNN, self).__init__(module=Net(self.device), proc_inputs=proc_inputs, proc_outputs=proc_outputs,
                                  *args, **kwargs)


class CSSM(ModuleWrapper):
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = F.tanh,
                 project_every: int = 0, local: bool = False, batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A = torch.nn.Linear(h_dim, h_dim, bias=False)
                self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False)
                self.C = torch.nn.Linear(h_dim, y_dim, bias=False)
                self.register_buffer('h_init', torch.randn((batch_size, h_dim)))
                self.register_buffer('h_next', torch.randn((batch_size, h_dim)))
                self.h = None
                self.dh = None
                self.sigma = sigma
                self.u_dim = u_dim
                self.du_dim = du_dim
                self.batch_size = batch_size
                self.delta = 1.
                self.local = local
                self.forward_count = 0
                self.project_every = project_every

            @torch.no_grad()
            def adjust_eigs(self):
                pass

            # noinspection PyUnusedLocal
            def init_h(self, udu: torch.Tensor) -> torch.Tensor:
                return self.h_init.data

            @staticmethod
            def handle_inputs(du, u):
                return du, u

            def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True):
                device = self.h_init.device
                u = u.flatten(1).to(device) if u is not None else torch.zeros((self.batch_size, self.u_dim),
                                                                              device=device)
                du = du.to(device) if du is not None else torch.zeros((self.batch_size, self.du_dim), device=device)

                if first:
                    h = self.init_h(torch.cat([du, u], dim=1))
                    self.forward_count = 0
                else:
                    h = self.h_next.data
                h.requires_grad_()

                if self.project_every:
                    if self.forward_count % self.project_every == 0:
                        self.adjust_eigs()

                du, u = self.handle_inputs(du, u)
                h_new = self.A(h) + self.B(torch.cat([du, u], dim=1))

                if self.local:
                    self.h = h
                    self.dh = (h_new - self.h) / self.delta
                else:
                    self.h = h_new
                    self.dh = (self.h - h) / self.delta

                y = self.C(self.sigma(self.h))
                self.h_next.data = h_new.detach()
                self.forward_count += 1
                return y

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)
        super(CSSM, self).__init__(module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs,
                                   *args, **kwargs)


class CDiagR(ModuleWrapper):
    """Diagonal matrix-based generator with real-valued transformations."""

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = lambda x: x,
                 project_every: int = 0, local: bool = False, batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, dtype=torch.float32)
                self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False)
                self.C = torch.nn.Linear(h_dim, y_dim, bias=False)
                self.register_buffer('h_init', torch.randn((batch_size, h_dim)))
                self.register_buffer('h_next', torch.randn((batch_size, h_dim)))
                self.h = None
                self.dh = None
                self.sigma = sigma
                self.u_dim = u_dim
                self.du_dim = du_dim
                self.batch_size = batch_size
                self.delta = 1.
                self.local = local
                self.forward_count = 0
                self.project_every = project_every

            @torch.no_grad()
            def adjust_eigs(self):
                self.diag.weight.copy_(torch.sign(self.diag.weight))

            # noinspection PyUnusedLocal
            def init_h(self, udu: torch.Tensor) -> torch.Tensor:
                return self.h_init.data

            @staticmethod
            def handle_inputs(du, u):
                return du, u

            def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True):
                device = self.h_init.device
                u = u.flatten(1).to(device) if u is not None else torch.zeros((self.batch_size, self.u_dim),
                                                                              device=device)
                du = du.to(device) if du is not None else torch.zeros((self.batch_size, self.du_dim), device=device)

                if first:
                    h = self.init_h(torch.cat([du, u], dim=1))
                    self.forward_count = 0
                else:
                    h = self.h_next.data
                h.requires_grad_()

                if self.project_every:
                    if self.forward_count % self.project_every == 0:
                        self.adjust_eigs()

                du, u = self.handle_inputs(du, u)
                h_new = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

                if self.local:
                    self.h = h
                    self.dh = (h_new - self.h) / self.delta
                else:
                    self.h = h_new
                    self.dh = (self.h - h) / self.delta

                y = self.C(self.sigma(self.h))
                self.h_next.data = h_new.detach()
                self.forward_count += 1
                return y

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)
        super(CDiagR, self).__init__(module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs,
                                     *args, **kwargs)


class CDiagC(ModuleWrapper):
    """Diagonal matrix-based generator with complex-valued transformations."""

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = lambda x: x,
                 project_every: int = 0, local: bool = False, batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, dtype=torch.cfloat)
                self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, dtype=torch.cfloat)
                self.C = torch.nn.Linear(h_dim, y_dim, bias=False, dtype=torch.cfloat)
                self.register_buffer('h_init', torch.randn((batch_size, h_dim)))
                self.register_buffer('h_next', torch.randn((batch_size, h_dim)))
                self.h = None
                self.dh = None
                self.sigma = sigma
                self.u_dim = u_dim
                self.du_dim = du_dim
                self.batch_size = batch_size
                self.delta = 1.
                self.local = local
                self.forward_count = 0
                self.project_every = project_every

            @torch.no_grad()
            def adjust_eigs(self):
                self.diag.weight.div_(self.diag.weight.abs())

            # noinspection PyUnusedLocal
            def init_h(self, udu: torch.Tensor) -> torch.Tensor:
                return self.h_init.data

            @staticmethod
            def handle_inputs(du, u):
                return du, u

            def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True):
                device = self.h_init.device
                u = u.flatten(1).to(device) if u is not None else torch.zeros((self.batch_size, self.u_dim),
                                                                              device=device, dtype=torch.cfloat)
                du = du.to(device) if du is not None else torch.zeros((self.batch_size, self.du_dim),
                                                                      device=device, dtype=torch.cfloat)

                if first:
                    h = self.init_h(torch.cat([du, u], dim=1))
                    self.forward_count = 0
                else:
                    h = self.h_next.data
                h.requires_grad_()

                if self.project_every:
                    if self.forward_count % self.project_every == 0:
                        self.adjust_eigs()

                du, u = self.handle_inputs(du, u)
                h_new = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

                if self.local:
                    self.h = h
                    self.dh = (h_new - self.h) / self.delta
                else:
                    self.h = h_new
                    self.dh = (self.h - h) / self.delta

                y = self.C(self.sigma(self.h))
                self.h_next.data = h_new.detach()
                self.forward_count += 1
                return y.real

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)
        super(CDiagC, self).__init__(module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs,
                                     *args, **kwargs)


class _CTENet(torch.nn.Module):
    """Inner stateful module for CTE-family wrappers (antisymmetric matrix-exp dynamics).

    Subclasses override ``init_h`` / ``handle_inputs`` to specialize the dynamics (e.g. zero-input
    forced-state variants) without rewriting the forward pass.
    """

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable, project_every: int, local: bool, cnu_memories: int, batch_size: int):
        super().__init__()
        self.W = torch.nn.Linear(h_dim, h_dim, bias=False)
        self.register_buffer('Id', torch.eye(h_dim))
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False)
        if cnu_memories <= 0:
            self.C = torch.nn.Linear(h_dim, y_dim, bias=False)
        else:
            self.C = LinearCNU(h_dim, y_dim, bias=False, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)
        self.register_buffer('h_init', torch.randn((batch_size, h_dim)))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim)))
        self.h = None
        self.dh = None
        self.sigma = sigma
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.batch_size = batch_size
        self.delta = delta
        self.local = local
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor | None, du: torch.Tensor | None, first: bool = True,
                last: bool = False) -> torch.Tensor:
        device = self.h_init.device
        u = u.flatten(1).to(device) if u is not None else torch.zeros((self.batch_size, self.u_dim), device=device)
        du = du.to(device) if du is not None else torch.zeros((self.batch_size, self.du_dim), device=device)

        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data
        h.requires_grad_()

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

        du, u = self.handle_inputs(du, u)
        A = 0.5 * (self.W.weight - self.W.weight.t())
        A_expm = torch.linalg.matrix_exp(A * self.delta)
        rec = F.linear(h, A_expm, self.W.bias)
        A_inv = torch.linalg.inv(A)
        inp = A_inv @ (A_expm - self.Id) @ self.B(torch.cat([du, u], dim=1)).unsqueeze(-1)

        h_new = rec + inp.squeeze(-1)
        if self.local:
            self.h = h
            self.dh = (h_new - self.h) / self.delta
        else:
            self.h = h_new
            self.dh = (self.h - h) / self.delta

        y = C(self.sigma(self.h))
        self.h_next.data = h_new
        self.forward_count += 1
        return y


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
    """

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 cnu_memories: int = 0, batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)
        super(CTE, self).__init__(
            module=_CTENet(u_dim, du_dim, y_dim, h_dim, delta, sigma, project_every, local,
                           cnu_memories, batch_size),
            proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class CTEInitStateBZeroInput(ModuleWrapper):

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 cnu_memories: int = 0, batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        class Net(_CTENet):
            @torch.no_grad()
            def init_h(self, udu: torch.Tensor) -> torch.Tensor:
                return self.B(udu).detach() / torch.sum(udu, dim=1)

            @staticmethod
            def handle_inputs(du, u):
                return torch.zeros_like(du), torch.zeros_like(u)

        super(CTEInitStateBZeroInput, self).__init__(
            module=Net(u_dim, du_dim, y_dim, h_dim, delta, sigma, project_every, local,
                       cnu_memories, batch_size),
            proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class CTEToken(ModuleWrapper):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int, y_dim: int, h_dim: int, *args, **kwargs):
        u_shape = torch.Size((emb_dim,))
        u_dim = u_shape.numel()
        du_dim = d_dim
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        class Net(_CTENet):
            def __init__(self):
                super().__init__(u_dim, du_dim, y_dim, h_dim, delta=1.0, sigma=lambda x: x,
                                 project_every=0, local=False, cnu_memories=0, batch_size=1)
                self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

            def forward(self, u: torch.Tensor | None, du: torch.Tensor | None,
                        first: bool = True, last: bool = False) -> torch.Tensor:
                if u is not None:
                    u = self.embeddings(u.to(self.embeddings.weight.device))
                return super().forward(u, du, first=first, last=last)

        super(CTEToken, self).__init__(
            module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


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

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float = 0.1,
                 alpha: float = 0., sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.order = h_dim // 2
                self.omega = torch.nn.Parameter(torch.empty(self.order))
                self.register_buffer('ones', torch.ones(self.order, requires_grad=False))
                self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False)
                self.C = torch.nn.Linear(h_dim, y_dim, bias=False)

                if alpha > 0.:
                    self.project_method = 'const'
                    self.register_buffer('alpha', torch.full_like(self.omega.data, alpha))
                elif alpha == 0.:
                    self.project_method = 'modulus'
                    self.register_buffer('alpha', torch.zeros_like(self.omega.data))
                elif alpha == -1.:
                    self.project_method = 'alpha'
                    self.register_buffer('alpha', torch.zeros_like(self.omega.data))

                self.register_buffer('h_init', torch.randn((batch_size, h_dim)))
                self.register_buffer('h_next', torch.randn((batch_size, h_dim)))
                self.h = None
                self.dh = None
                self.sigma = sigma
                self.u_dim = u_dim
                self.du_dim = du_dim
                self.batch_size = batch_size
                self.delta = delta
                self.local = local
                self.forward_count = 0
                self.project_every = project_every
                self.reset_parameters()

            def reset_parameters(self) -> None:
                torch.nn.init.uniform_(self.omega)

            @torch.no_grad()
            def adjust_eigs(self):
                if self.project_method == 'alpha':
                    self.alpha.copy_((1. - torch.sqrt(1. - (self.delta * self.omega) ** 2) / self.delta))
                elif self.project_method == 'modulus':
                    module = torch.sqrt(self.ones ** 2 + (self.delta * self.omega) ** 2)
                    self.omega.div_(module)
                    self.ones.div_(module)

            # noinspection PyUnusedLocal
            def init_h(self, udu: torch.Tensor) -> torch.Tensor:
                return self.h_init.data

            @staticmethod
            def handle_inputs(du, u):
                return du, u

            def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True):
                device = self.h_init.device
                u = u.flatten(1).to(device) if u is not None else torch.zeros((self.batch_size, self.u_dim),
                                                                              device=device)
                du = du.to(device) if du is not None else torch.zeros((self.batch_size, self.du_dim), device=device)

                if first:
                    h = self.init_h(torch.cat([du, u], dim=1))
                    self.forward_count = 0
                else:
                    h = self.h_next.data
                h.requires_grad_()
                h_pair = h.view(-1, self.order, 2)

                if self.project_every:
                    if self.forward_count % self.project_every == 0:
                        self.adjust_eigs()

                du, u = self.handle_inputs(du, u)
                h1 = (self.ones - self.delta * self.alpha) * h_pair[..., 0] + self.delta * self.omega * h_pair[..., 1]
                h2 = -self.delta * self.omega * h_pair[..., 0] + (self.ones - self.delta * self.alpha) * h_pair[..., 1]
                rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)
                inp = self.delta * self.B(torch.cat([du, u], dim=1))

                h_new = rec + inp
                if self.local:
                    self.h = h
                    self.dh = (h_new - self.h) / self.delta
                else:
                    self.h = h_new
                    self.dh = (self.h - h) / self.delta

                y = self.C(self.sigma(self.h))
                self.h_next.data = h_new.detach()
                self.forward_count += 1
                return y

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)
        super(CTB, self).__init__(module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs,
                                  *args, **kwargs)


class _CTBENet(torch.nn.Module):
    """Inner stateful module for CTBE-family wrappers (exact-rotation antisymmetric blocks).

    Subclasses override ``init_h`` / ``handle_inputs`` to specialize the dynamics without
    rewriting the forward pass.
    """

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable, project_every: int, local: bool, cnu_memories: int, batch_size: int):
        super().__init__()
        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"
        self.order = h_dim // 2
        self.omega = torch.nn.Parameter(torch.empty(self.order))
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False)
        if cnu_memories <= 0:
            self.C = torch.nn.Linear(h_dim, y_dim, bias=False)
        else:
            self.C = LinearCNU(h_dim, y_dim, bias=False, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)
        self.register_buffer('h_init', torch.randn((batch_size, h_dim)))
        self.register_buffer('h_next', torch.randn((batch_size, h_dim)))
        self.h = None
        self.dh = None
        self.sigma = sigma
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.batch_size = batch_size
        self.delta = delta
        self.local = local
        self.forward_count = 0
        self.project_every = project_every
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not isinstance(self.omega, CNUs):
            torch.nn.init.uniform_(self.omega)
        else:
            torch.nn.init.uniform_(self.omega.M)

    @torch.no_grad()
    def adjust_eigs(self):
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init.data

    @staticmethod
    def handle_inputs(du, u):
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = True) -> torch.Tensor:
        device = self.h_init.device
        u = u.flatten(1).to(device) if u is not None else torch.zeros((self.batch_size, self.u_dim), device=device)
        du = du.to(device) if du is not None else torch.zeros((self.batch_size, self.du_dim), device=device)

        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next.data
        h.requires_grad_()
        h_pair = h.view(-1, self.order, 2)

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

        du, u = self.handle_inputs(du, u)
        udu = torch.cat([du, u], dim=1)
        cos_t = torch.cos(self.omega * self.delta)
        sin_t = torch.sin(self.omega * self.delta)

        h1 = cos_t * h_pair[..., 0] + sin_t * h_pair[..., 1]
        h2 = -sin_t * h_pair[..., 0] + cos_t * h_pair[..., 1]
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)

        u_hat = self.B(udu).view(-1, self.order, 2)
        inp1 = (sin_t * u_hat[..., 0] - (cos_t - 1) * u_hat[..., 1]) / self.omega
        inp2 = ((cos_t - 1) * u_hat[..., 0] + sin_t * u_hat[..., 1]) / self.omega
        inp = torch.stack([inp1, inp2], dim=-1).flatten(start_dim=1)

        h_new = rec + inp
        if self.local:
            self.h = h
            self.dh = (h_new - self.h) / self.delta
        else:
            self.h = h_new
            self.dh = (self.h - h) / self.delta

        y = C(self.sigma(self.h))
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
                 cnu_memories: int = 0, batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)
        super(CTBE, self).__init__(
            module=_CTBENet(u_dim, du_dim, y_dim, h_dim, delta, sigma, project_every, local,
                            cnu_memories, batch_size),
            proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class CTBEInitStateBZeroInput(ModuleWrapper):

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 cnu_memories: int = 0, batch_size: int = 1, *args, **kwargs):
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_rnn(u_shape, du_dim, y_dim)

        class Net(_CTBENet):
            @torch.no_grad()
            def init_h(self, udu: torch.Tensor) -> torch.Tensor:
                return self.B(udu).detach() / torch.sum(udu)

            @staticmethod
            def handle_inputs(du, u):
                return torch.zeros_like(du), torch.zeros_like(u)

        super(CTBEInitStateBZeroInput, self).__init__(
            module=Net(u_dim, du_dim, y_dim, h_dim, delta, sigma, project_every, local,
                       cnu_memories, batch_size),
            proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class CNN(ModuleWrapper):
    def __init__(self, d_dim: int, in_channels: int = 3, in_res: int = 32, *args, **kwargs):
        net = torch.nn.Sequential(
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
            torch.nn.Sigmoid())

        transforms = transforms_factory("rgb" + str(in_res) if in_channels == 3 else "gray" + str(in_res))
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim, transforms)
        super(CNN, self).__init__(net, proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class CNNCNU(ModuleWrapper):
    def __init__(self, d_dim: int, cnu_memories: int, in_channels: int = 3, in_res: int = 32,
                 delta: int = 1, scramble: bool = False, *args, **kwargs):
        net = torch.nn.Sequential(
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
            torch.nn.Sigmoid())

        transforms = transforms_factory("rgb" + str(in_res) if in_channels == 3 else "gray" + str(in_res))
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim, transforms)
        super(CNNCNU, self).__init__(net, proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class SingleLayerCNU(ModuleWrapper):
    def __init__(self, d_dim: int, cnu_memories: int, in_channels: int = 3, in_res: int = 32,
                 delta: int = 1, scramble: bool = False, *args, **kwargs):
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            LinearCNU(in_res * in_res * in_channels, d_dim, key_mem_units=cnu_memories, delta=delta, scramble=scramble),
            torch.nn.Sigmoid())

        transforms = transforms_factory("rgb" + str(in_res) if in_channels == 3 else "gray" + str(in_res))
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim, transforms)
        super(SingleLayerCNU, self).__init__(net, proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class CNNMNIST(CNN):
    def __init__(self, *args, **kwargs):
        kwargs['in_channels'] = 1
        kwargs['in_res'] = 28
        super(CNNMNIST, self).__init__(*args, **kwargs)
        for p in self.proc_inputs:
            for prop in p.props:
                prop.set_stream_to_proc_transforms(transforms_factory("gray_mnist"))


class CNNCNUMNIST(CNNCNU):
    def __init__(self, *args, **kwargs):
        kwargs['in_channels'] = 1
        kwargs['in_res'] = 28
        super(CNNCNUMNIST, self).__init__(*args, **kwargs)
        for p in self.proc_inputs:
            for prop in p.props:
                prop.set_stream_to_proc_transforms(transforms_factory("gray_mnist"))


class SingleLayerCNUMNIST(SingleLayerCNU):
    def __init__(self, *args, **kwargs):
        kwargs['in_channels'] = 1
        kwargs['in_res'] = 28
        super(SingleLayerCNUMNIST, self).__init__(*args, **kwargs)
        for p in self.proc_inputs:
            for prop in p.props:
                prop.set_stream_to_proc_transforms(transforms_factory("gray_mnist"))


class ResNet(ModuleWrapper):
    def __init__(self, d_dim: int = -1, freeze_backbone: bool = True, *args, **kwargs):
        net = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
            for p in net.fc.parameters():
                p.requires_grad = True

        if d_dim > 0:
            net.fc = torch.nn.Sequential(
                torch.nn.Linear(net.fc.in_features, d_dim),
                torch.nn.Sigmoid())

        transforms = transforms_factory("rgb224")
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim, transforms)
        super(ResNet, self).__init__(net, proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class ResNetCNU(ModuleWrapper):
    def __init__(self, d_dim: int, cnu_memories: int,
                 delta: int = 1, scramble: bool = False, freeze_backbone: bool = True, *args, **kwargs):
        net = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
            for p in net.fc.parameters():
                p.requires_grad = True

        if d_dim > 0:
            net.fc = torch.nn.Sequential(
                LinearCNU(net.fc.in_features, d_dim, key_mem_units=cnu_memories, delta=delta, scramble=scramble),
                torch.nn.Sigmoid())

        transforms = transforms_factory("rgb224")
        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim, transforms)
        super(ResNetCNU, self).__init__(net, proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class ViT(ModuleWrapper):
    def __init__(self, d_dim: int = -1, *args, **kwargs):
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        transforms = torchvision.transforms.Compose([
            weights.transforms(),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
        ])
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
            with urllib.request.urlopen(url) as f:
                self.labels = [line.strip().decode('utf-8') for line in f.readlines()]

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = vit
                self.tfm = transforms

            def forward(self, y: Image.Image):
                device = next(self.backbone.parameters()).device
                return self.backbone(self.tfm(y).to(device))

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        super(ViT, self).__init__(module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs, *args, **kwargs)


class DenseNet(ModuleWrapper):
    def __init__(self, d_dim: int = -1, *args, **kwargs):
        transforms = transforms_factory("rgb224")
        densenet = torchvision.models.densenet121(weights=None)

        if d_dim > 0:
            densenet.classifier = torch.nn.Sequential(
                torch.nn.Linear(densenet.classifier.in_features, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, d_dim),
                torch.nn.Sigmoid()
            )

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = densenet
                self.tfm = transforms

            def forward(self, y: Image.Image):
                device = next(self.backbone.parameters()).device
                return self.backbone(self.tfm(y).to(device))

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        super(DenseNet, self).__init__(module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs,
                                       *args, **kwargs)


class EfficientNet(ModuleWrapper):
    def __init__(self, d_dim: int = -1, *args, **kwargs):
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        transforms = torchvision.transforms.Compose([
            weights.transforms(),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
        ])
        effnet = torchvision.models.efficientnet_b0(weights=weights)

        if d_dim > 0:
            effnet.classifier = torch.nn.Sequential(
                torch.nn.Linear(effnet.classifier[1].in_features, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, d_dim),
                torch.nn.Sigmoid()
            )

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = effnet
                self.tfm = transforms

            def forward(self, y: Image.Image):
                device = next(self.backbone.parameters()).device
                o = self.backbone(self.tfm(y).to(device))
                if o.dim() == 1:
                    o = o.unsqueeze(0)
                return o

        proc_inputs, proc_outputs = get_proc_inputs_and_proc_outputs_for_image_classification(d_dim)
        super(EfficientNet, self).__init__(module=Net(), proc_inputs=proc_inputs, proc_outputs=proc_outputs,
                                           *args, **kwargs)


class FasterRCNN(ModuleWrapper):
    def __init__(self, *args, **kwargs):
        self.labels: list[str] = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                                  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                  'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                                  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                  'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                                  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                                  'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                                  'cell phone',
                                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                                  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                                  ]
        labels = self.labels

        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        faster_rcnn.eval()
        transforms = torchvision.transforms.Compose([transforms_factory("rgb-no_norm"),
                                                     torchvision.transforms.Lambda(lambda x: x.squeeze(0)),
                                                     weights.transforms()])

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = faster_rcnn
                self.tfm = transforms
                self.labels = labels

            def forward(self, y: Image.Image):
                device = next(self.backbone.parameters()).device
                o = self.backbone([self.tfm(y).to(device)])  # List with 1 image per element (no batch dim)

                found_class_indices = o[0]['labels']
                found_class_scores = o[0]['scores']
                found_class_boxes = o[0]['boxes']
                valid = found_class_scores > 0.8

                found_class_indices: torch.Tensor = found_class_indices[valid]
                found_class_scores = found_class_scores[valid]
                found_class_boxes = found_class_boxes[valid]
                found_class_names: list[str] = [self.labels[int(i.item())] for i in found_class_indices]

                return found_class_indices, found_class_scores, found_class_boxes, ", ".join(found_class_names)

        super(FasterRCNN, self).__init__(
            module=Net(),
            proc_inputs=[StreamType(data_type="img", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="tensor", tensor_dtype=torch.long, tensor_shape=(None,),
                                     pubsub=False, private_only=False),
                          StreamType(data_type="tensor", tensor_dtype=torch.float32, tensor_shape=(None,),
                                     pubsub=False, private_only=False),
                          StreamType(data_type="tensor", tensor_dtype=torch.float32, tensor_shape=(None, 4),
                                     pubsub=False, private_only=False),
                          StreamType(data_type="text",
                                     pubsub=False, private_only=False)],
            *args, **kwargs)


class TinyLLama(ModuleWrapper):
    def __init__(self, device=None, *args, **kwargs):

        class Net(torch.nn.Module):
            def __init__(self, _device):
                super().__init__()
                self.__pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                       torch_dtype=torch.bfloat16, device=_device)

            def forward(self, msg: str) -> str:
                msg_struct = [{"role": "system", "content": "You are a helpful assistant"},
                              {"role": "user", "content": msg}]
                assert self.__pipe.tokenizer is not None
                prompt = self.__pipe.tokenizer.apply_chat_template(msg_struct, tokenize=False,
                                                                   add_generation_prompt=True)
                assert isinstance(prompt, str)
                out: list = self.__pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7,
                                        top_k=50, top_p=0.95)
                out: str = out[0]["generated_text"] if (out is not None and len(out) > 0 and
                                                        "generated_text" in out[0]) else "Error!"
                if "<|assistant|>\n" in out:
                    out = out.split("<|assistant|>\n")[1]
                return out.strip()

        # Populate self.device
        self.guess_device(device)

        super(TinyLLama, self).__init__(
            module=Net(self.device),
            device=device,
            proc_inputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            *args, **kwargs
        )


class LLama(ModuleWrapper):
    def __init__(self, device=None, *args, **kwargs):

        class Net(torch.nn.Module):
            def __init__(self, _device):
                super().__init__()
                self.__pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct",
                                       torch_dtype=torch.bfloat16, device=_device)

            def forward(self, msg: str) -> str:
                msg_struct = [{"role": "system", "content": "You are a helpful assistant"},
                              {"role": "user", "content": msg}]
                assert self.__pipe.tokenizer is not None
                prompt = self.__pipe.tokenizer.apply_chat_template(msg_struct, tokenize=False,
                                                                   add_generation_prompt=True)
                assert isinstance(prompt, str)
                out = self.__pipe(prompt, max_new_tokens=256, do_sample=True, return_full_text=False,
                                  temperature=0.7, top_k=50, top_p=0.95)
                out = out[0]["generated_text"] if (out is not None and len(out) > 0 and
                                                   "generated_text" in out[0]) else "Error!"
                if "<|assistant|>\n" in out:
                    out = out.split("<|assistant|>\n")[1]
                return out.strip()

        # Populate self.device
        self.guess_device(device)

        super(LLama, self).__init__(
            module=Net(self.device),
            device=device,
            proc_inputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            *args, **kwargs
        )


class Phi(ModuleWrapper):
    def __init__(self, device=None, *args, **kwargs):
        class Net(torch.nn.Module):
            def __init__(self, _device):
                super().__init__()
                self.__pipe = pipeline("text-generation", model="microsoft/Phi-3.5-mini-instruct",
                                       torch_dtype="auto", device=_device)

            def forward(self, msg: str) -> str:
                msg_struct = [{"role": "system", "content": "You are a helpful assistant"},
                              {"role": "user", "content": msg}]
                assert self.__pipe.tokenizer is not None
                prompt = self.__pipe.tokenizer.apply_chat_template(msg_struct, tokenize=False,
                                                                   add_generation_prompt=True)
                assert isinstance(prompt, str)
                out_: list = self.__pipe(prompt, max_new_tokens=256, do_sample=True, return_full_text=False)
                out: str = out_[0]["generated_text"] if (out_ is not None and len(out_) > 0 and
                                                         "generated_text" in out_[0]) else "Error!"
                if "<|assistant|>\n" in out:
                    out = out.split("<|assistant|>\n")[1]
                return out.strip()

        # Populate self.device
        self.guess_device(device)

        super(Phi, self).__init__(
            module=Net(self.device),
            device=device,
            proc_inputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            *args, **kwargs
        )


class LangSegmentAnything(ModuleWrapper):
    def __init__(self, device=None, *args, **kwargs):
        from lang_sam import LangSAM
        from PIL import ImageDraw, ImageFont

        class Net(torch.nn.Module):
            def __init__(self, _device):
                super().__init__()

                # Generate a 64x64 error image (with text "Error" on it)
                error_img = Image.new("RGB", (64, 64), color="white")
                draw = ImageDraw.Draw(error_img)
                font = ImageFont.load_default()
                text = "Error"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                position = ((64 - text_width) // 2, (64 - text_height) // 2)
                draw.text(position, text, fill="black", font=font)

                self.__sam = LangSAM(device=_device)
                self.__error_img = error_img

            def forward(self, image_pil: Image.Image, msg: str):
                try:
                    image_pil = image_pil.convert("RGB") if image_pil.mode != "RGB" else image_pil  # Forcing RGB
                    out = self.__sam.predict([image_pil], [msg])
                    if (out is None or not isinstance(out, list) or len(out) < 1 or not isinstance(out[0], dict) or
                            'masks' not in out[0]) or out[0]['masks'].ndim != 3:
                        return image_pil
                    else:
                        return LangSegmentAnything.highlight_masks_on_image(image_pil, out[0]['masks'])
                except Exception:
                    return self.__error_img

        # Populate self.device
        self.guess_device(device)

        super(LangSegmentAnything, self).__init__(
            module=Net(self.device),
            device=device,
            proc_inputs=[StreamType(data_type="img", pubsub=False, private_only=False),
                         StreamType(data_type="text", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="img", pubsub=False, private_only=False)],
            *args, **kwargs
        )

    @staticmethod
    def highlight_masks_on_image(image_pil: Image.Image, masks: np.ndarray, alpha: float = 0.75):
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
    def __init__(self, device=None, *args, **kwargs):
        from transformers import AutoModelForImageTextToText

        class Net(torch.nn.Module):
            def __init__(self, _device):
                super().__init__()
                model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
                self.__backbone = (
                    AutoModelForImageTextToText.from_pretrained(model_id,
                                                                torch_dtype=torch.bfloat16,
                                                                device_map=_device).to(_device))
                self.__pp = AutoProcessor.from_pretrained(model_id, device_map=_device)

            def forward(self, image_pil: Image.Image, msg: str = "what is this?"):
                image_pil = image_pil.convert("RGB") if image_pil.mode != "RGB" else image_pil  # Forcing RGB
                _device = next(self.__backbone.parameters()).device

                msg_struct = [{"role": "user", "content": [{"type": "text", "text": f"{msg}"},
                                                           {"type": "image", "image": image_pil}]}]

                prompt = self.__pp.apply_chat_template(msg_struct,
                                                       tokenize=True,
                                                       add_generation_prompt=True,
                                                       return_dict=True,
                                                       return_tensors="pt").to(_device, dtype=torch.bfloat16)

                out = self.__backbone.generate(**prompt, do_sample=False, max_new_tokens=128)
                out = self.__pp.batch_decode(out, skip_special_tokens=True)[0] if out is not None else "Error!"
                if "Assistant:" in out:
                    out = out.split("Assistant:")[1]
                return out.strip()

        # Populate self.device
        self.guess_device(device)

        super(SmolVLM, self).__init__(
            module=Net(self.device),
            device=device,
            proc_inputs=[StreamType(data_type="img", pubsub=False, private_only=False),
                         StreamType(data_type="text", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            *args, **kwargs
        )


class SiteRAG(ModuleWrapper):

    def __init__(self,
                 site_url: str,
                 site_folder: str = os.path.join("rag", "downloaded_site"),
                 db_folder: str = os.path.join("rag", "chroma_db"),
                 *args, **kwargs):
        # Saving options
        self.site_url = site_url
        self.site_folder = site_folder
        self.db_folder = db_folder

        # Loading neural model
        device_env = os.getenv("PROC_DEVICE", None)
        target_device = torch.device("cpu") if device_env is None else torch.device(device_env)
        model_id = "TheBloke/vicuna-7b-1.1-HF"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                                     device_map=target_device, offload_folder="offload")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

        # Embedder
        from langchain.embeddings import SentenceTransformerEmbeddings
        self.embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2",
                                                      model_kwargs={"device": target_device.type})

        # Crawling site (uses self.embedder + self.site_folder + self.db_folder + self.site_url)
        self.crawl_website()
        self.crawled_site_to_rag_knowledge_base()

        # Setting up RAG stuff
        from langchain.vectorstores import Chroma
        db = Chroma(persist_directory=db_folder, embedding_function=self.embedder)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._pipe = pipe
                self._retriever = retriever

            def forward(self, msg: str):
                # Build context
                docs = self._retriever.get_relevant_documents(msg)
                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {msg}\nAnswer:"

                # Generate answer
                out = self._pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
                out = out[0]['generated_text'][len(prompt):].strip() if (out is not None and len(out) > 0 and
                                                                         "generated_text" in out[0]) else "Error!"

                # Append source URLs
                best_doc_with_score = self._retriever.vectorstore.similarity_search_with_score(msg, k=1)
                best_doc, _ = best_doc_with_score[0]
                docs = [best_doc]
                sources = set("<a href='" +
                              doc.metadata['source'] +
                              "' onclick='window.open(this.href); return false;' style='color: blue;'>" +
                              doc.metadata['source'] + "</a>" for doc in docs)
                sources_text = "<br/><br/>\nURLs:\n" + "\n".join(sources)

                return out.strip() + sources_text

        super(SiteRAG, self).__init__(
            module=Net(),
            proc_inputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            *args, **kwargs
        )

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
                    link: dict
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


class FeatherlessAPI(ModuleWrapper):
    """Callable handle onto the shared Featherless gateway.

    Typical usage::

        api = FeatherlessAPI(model="some-model-id", cost=2)
        text = api("write me a haiku")  # Routed through the gateway

    One instance per logical caller. The model and unit cost are fixed at construction, so the callable takes only the
    prompt string. Construction bootstraps the shared server if needed (self-spawning, race-safe) and opens this
    caller's persistent registration socket: the liveness token whose lifetime equals this object's interest in the
    gateway. Closing the instance (or letting the process die) releases it; when the last instance goes away the server
    shuts itself down.

    The whole client lifecycle (bootstrap, registration, request round-trip) is self-contained here; callers never
    touch the server internals.
    """

    def __init__(self, model: str | None = None, cost: int = 1, system_prompt: str = "",
                 process_id: str | None = None, max_tokens: int = -1, temperature: float = -1.,
                 top_p: float = -1., top_k: int = -1, frequency_penalty: float | None = None,
                 presence_penalty: float | None = None, repetition_penalty: float | None = None,
                 min_p: float | None = None, sampler: dict | None = None, connect_timeout: float = 15.0,
                 *args, **kwargs):
        """Create a FeatherlessAPI handle and connect it to the shared gateway.

        Args:
            model: The model identifier used for every call (None lets the server fall back to its MODEL_ID default).
            cost: The unit cost charged for every call (one of VALID_COSTS) (Default: 1).
            system_prompt: The system prompt prepended to every call ("" means no system prompt) (Default: "").
            process_id: Identifier used for round-robin fairness; defaults to this process's PID.
            max_tokens: Maximum number of tokens to generate per call (-1 means no limit) (Default: -1).
            temperature: Sampling temperature for every call (negative lets the API use its default) (Default: -1.).
            top_p: Nucleus-sampling probability (negative lets the API use its default) (Default: -1.).
            top_k: Top-k sampling cutoff (negative lets the API use its default) (Default: -1).
            frequency_penalty: Frequency penalty (None lets the API use its default) (Default: None).
            presence_penalty: Presence penalty (None lets the API use its default) (Default: None).
            repetition_penalty: Repetition penalty, a vLLM/Featherless extension (None uses the default) (Default None).
            min_p: Minimum-probability cutoff, a vLLM/Featherless extension (None uses the default) (Default: None).
            sampler: Extra sampler params merged last (its keys win); use it for any knob not covered above.
            connect_timeout: Maximum seconds to wait for the gateway server to come up (Default: 15.0).
        """
        if cost not in APIGatewayServer.VALID_COSTS:
            log.critical(f"Invalid cost {cost}: it must be one of {APIGatewayServer.VALID_COSTS}")

        # Bring up the shared gateway server before opening any sockets (idempotent across processes)
        FeatherlessAPI._ensure_server(connect_timeout)

        class Net(torch.nn.Module):
            """Holds the sampler config and the two gateway sockets, and routes each `forward(prompt)`
            through the gateway. Plain Python attributes (socket, dict, str): none get registered as
            torch submodules since none are `nn.Module`/`Parameter`/`Tensor`."""

            def __init__(self):
                super().__init__()
                self.model_name: str | None = model
                self.cost: int = cost
                self.system_prompt: str = system_prompt

                # Per-call sampler: include each knob only when explicitly set, so unset ones fall back
                # to the API default. The free-form `sampler` arg is merged last and overrides.
                self.sampler: dict = {}
                if max_tokens > 0:
                    self.sampler["max_tokens"] = max_tokens
                if temperature >= 0.:
                    self.sampler["temperature"] = temperature
                if top_p >= 0.:
                    self.sampler["top_p"] = top_p
                if top_k >= 0:
                    self.sampler["top_k"] = top_k
                if frequency_penalty is not None:
                    self.sampler["frequency_penalty"] = frequency_penalty
                if presence_penalty is not None:
                    self.sampler["presence_penalty"] = presence_penalty
                if repetition_penalty is not None:
                    self.sampler["repetition_penalty"] = repetition_penalty
                if min_p is not None:
                    self.sampler["min_p"] = min_p
                if sampler:
                    self.sampler.update(sampler)

                # Round-robin fairness is per process; default ID is the PID
                self.process_id: str = str(process_id if process_id is not None else os.getpid())

                # Persistent registration socket: lifetime == this caller's interest in the gateway
                self._reg: socket.socket = socket.create_connection(
                    (APIGatewayServer.HOST, APIGatewayServer.PORT))
                self._reg.sendall(b'{"op":"hello"}\n')

                # Separate request socket (one in-flight request per instance; this client is synchronous)
                self._req: socket.socket = socket.create_connection(
                    (APIGatewayServer.HOST, APIGatewayServer.PORT))
                self._rf = self._req.makefile("r")

            def forward(self, prompt: str) -> str:
                if not isinstance(prompt, str):
                    log.critical(f"Invalid prompt: it must be a str, got {type(prompt).__name__}")
                msg = json.dumps({"op": "generate", "process_id": self.process_id,
                                  "sys_prompt": self.system_prompt, "prompt": prompt,
                                  "cost": self.cost, "model": self.model_name,
                                  "sampler": self.sampler}) + "\n"
                self._req.sendall(msg.encode())
                line = self._rf.readline()
                if not line:
                    log.critical("Gateway closed the connection")
                resp = json.loads(line)
                if not resp.get("ok"):
                    log.critical(f"Gateway returned an error: {resp.get('error', 'unknown error')}")
                return resp["result"]

            def close(self) -> None:
                """Close both gateway sockets, releasing this caller's interest."""
                for s in (self._req, self._reg):
                    try:
                        s.close()
                    except OSError:
                        pass

        super(FeatherlessAPI, self).__init__(
            module=Net(),
            proc_inputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            proc_outputs=[StreamType(data_type="text", pubsub=False, private_only=False)],
            *args, **kwargs
        )

    def close(self) -> None:
        """Close both the request and the persistent registration socket, releasing this caller's interest."""
        assert self.module is not None
        if hasattr(self.module, 'close'):
            self.module.close()

    def __enter__(self) -> 'FeatherlessAPI':
        """Enter the context manager, returning this handle."""
        return self

    def __exit__(self, *exc) -> None:
        """Exit the context manager, closing the handle."""
        self.close()

    @staticmethod
    def _server_is_up() -> bool:
        """Return True if a gateway server is already listening on the configured host/port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            return s.connect_ex((APIGatewayServer.HOST, APIGatewayServer.PORT)) == 0

    @staticmethod
    def _spawn_detached_server() -> None:
        """Launch the gateway server as a fully independent, detached process."""
        import subprocess

        # start_new_session=True puts the server in its own session/process group: it is not killed when we die, and
        # does not receive our Ctrl-C / SIGINT. The server lives in unaiverse.modules.utils; we call its
        # serve_api_gateway() function via -c rather than running the module with -m, because importing the parent
        # package already imports utils, so -m/runpy would execute a second copy of it under __main__ (RuntimeWarning
        # + duplicate classes).
        cmd = [sys.executable, "-c", "from unaiverse.modules.utils import serve_api_gateway; serve_api_gateway()"]
        subprocess.Popen(
            cmd,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
            # stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    @classmethod
    def _ensure_server(cls, timeout: float = 15.0) -> None:
        """Ensure exactly one gateway server is running, racing safely across processes.

        Fast path: if the port is already up, return immediately. Slow path: acquire a flock-based mutex (held only
        during the short spawn-and-wait critical section). The single holder re-checks the port and, if still down,
        spawns the detached server and waits for it to bind. Other processes block on the flock; by the time they
        acquire it the port is up, so they spawn nothing. A starter that crashes mid-spawn releases the flock
        automatically (it is OS-released on death), so there are no stale locks to clean.

        Args:
            timeout: Maximum seconds to wait for the server to become reachable (Default: 15.0).
        """
        import fcntl

        if cls._server_is_up():
            return

        deadline = time.time() + timeout

        # Open (not O_EXCL): the file is a mutex, not an election token
        lock_fd = os.open(APIGatewayServer.LOCKFILE, os.O_CREAT | os.O_RDWR)
        try:

            # Block until we hold the lock (or time out via the loop below)
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if cls._server_is_up():
                        return  # Someone else's spawn already succeeded
                    if time.time() > deadline:
                        log.critical("Timed out waiting for the gateway server to start up")
                    time.sleep(0.05)

            # We hold the mutex: re-check under the lock, since another holder may have just brought the server up
            if cls._server_is_up():
                return

            cls._spawn_detached_server()

            while time.time() < deadline:
                if cls._server_is_up():
                    return
                time.sleep(0.05)
            log.critical("Gateway server did not come up")
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
