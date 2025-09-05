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
import torch
import torch.nn as nn
from typing import Iterable, Dict, Any


def _euler_step(o: torch.Tensor | Dict[str, torch.Tensor], do: torch.Tensor | Dict[str, torch.Tensor],
                step_size: float, decay: float | None = None, in_place: bool = False) \
        -> torch.Tensor | Dict[str, torch.Tensor] | None:
    """Euler step, vanilla case.

    Params:
        o: list or dict of data to update (warning: it will be updated here).
        do: list or dict of derivatives w.r.t. time of the data we want to update.
        step_size: the step size of the Euler method.
        decay: the weight-decay-like scalar coefficient that tunes the strength of the weight-decay regularization.
        in_place: whether to overwrite the input data or to return a new list with new elements.

    Returns:
        A list or dict of Tensors with the same size of the input 'o'. It could be a new list with new Tensors
        (if in_place is False) or 'o' itself, updated in-place (if in_place is True).
    """
    assert type(o) is type(do), f'Input should either be two lists or two dicts, got {type(o)} and {type(do)}.'

    if isinstance(o, dict):
        assert set(o.keys()) == set(do.keys()), 'Dictionaries should have the same keys.'
        oo = dict.fromkeys(o)
        for k in o.keys():
            if not in_place:
                if decay is None or decay == 0.:
                    oo[k] = o[k] + step_size * do[k]
                else:
                    oo[k] = (1. - decay) * o[k] + step_size * do[k]
            else:
                if decay is None or decay == 0.:
                    o[k].add_(do[k], alpha=step_size)
                else:
                    o[k].mul_(1. - decay).add_(do[k], alpha=step_size)
    elif isinstance(o, torch.Tensor):
        if not in_place:
            if decay is None or decay == 0.:
                oo = o + step_size * do
            else:
                oo = (1. - decay) * o + step_size * do
        else:
            oo = None
            if decay is None or decay == 0.:
                o.add_(do, alpha=step_size)
            else:
                o.mul_(1. - decay).add_(do, alpha=step_size)
    else:
        raise Exception(f'Input to this function should be either tensor or dict, got {type(o)}.')

    if not in_place:
        return oo
    else:
        return o


def _init(val: float | str, data_shape: torch.Size, device, dtype: torch.dtype, non_negative: bool = False) \
        -> torch.Tensor:
    """Initialize a tensor to a constant value or to random values, or to zeros (and possibly others).

    Params:
        val: a float value or a string in ['random', 'zeros'].
        data_shape: the shape of the target tensor.
        device: the device where the tensor will be stored.
        non_negative: whether to create something non-negative.

    Returns:
        An initialized tensor.
    """
    assert type(val) is float or val in ['zeros', 'random', 'ones', 'alternating'], (
            'Invalid initialization: ' + str(val))

    if isinstance(val, float):
        t = torch.full(data_shape, val, device=device, dtype=dtype)
        if non_negative:
            t = torch.abs(t)
        return t
    elif val == 'random':
        t = torch.randn(data_shape, device=device, dtype=dtype)
        if non_negative:
            t = torch.abs(t)
        return t
    elif val == 'zeros':
        return torch.zeros(data_shape, device=device, dtype=dtype)
    elif val == 'ones':
        return torch.ones(data_shape, device=device, dtype=dtype)
    elif val == 'alternating':

        # Initialize the state as alternating pairs of (0,1) (to be used with BlockSkewSymmetric)
        # data_shape is (batch_size, xi_shape)
        assert len(data_shape) == 2, (f"xi should be initialized as (batch_size, xi_shape), "
                                      f"got xi with {len(data_shape)} dimensions.")
        order = data_shape[1] // 2
        batch_size = data_shape[0]
        return torch.tensor([[0., 1.]], device=device, dtype=dtype).repeat(batch_size, order)


def _init_state_and_costate(model: nn.Module, batch_size: int = 1) -> (
        Dict)[str, Dict[str, torch.Tensor | Dict[str, torch.Tensor]]]:
    """Initialize the state and costate dictionaries (keys are 'xi', 'w_xi', 'w_y').
    """

    # Getting device
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Creating state and costate
    x = dict(xi=model.h_init, w={})
    p = dict(xi=torch.zeros_like(x['xi'], device=device, dtype=dtype), w={})

    # Initialize state and costate for the weights of the state network
    x['w'] = {par_name: par for par_name, par in dict(model.named_parameters()).items() if par.requires_grad}
    p['w'] = {par_name: torch.zeros_like(par, device=device, dtype=dtype) for par_name, par in x['w'].items()}

    return {'x': x, 'p': p}


def _get_grad(a: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Collects gradients from a list of Tensors.

    Params:
        a: list or dict of Tensors.

    Returns:
        List or dict of references to the 'grad' fields of each component of the input list.
    """
    if isinstance(a, dict):
        g = {_k_a: a[_k_a].grad if a[_k_a].grad is not None else torch.zeros_like(a[_k_a]) for _k_a in a.keys()}
    elif isinstance(a, torch.Tensor):
        g = a.grad if a.grad is not None else torch.zeros_like(a)
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return g


def _apply1(a: torch.Tensor | Dict[str, torch.Tensor], op: torch.func) \
        -> torch.Tensor | Dict[str, torch.Tensor]:
    """Apply an operation to each element on a list or dict of Tensors.

    Params:
        a: list or dict of Tensors.
        op: operation to be applied to the elements in list_ten (it could be a lambda expression).

    Returns:
        List or dict of Tensors with the result of the operation (same size of each list-argument).
    """
    if isinstance(a, dict):
        oo = {_k_a: op(a[_k_a]) for _k_a in a.keys()}
    elif isinstance(a, torch.Tensor):
        oo = op(a)
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return oo


def _apply2(a: torch.Tensor | Dict[str, torch.Tensor], b: torch.Tensor | Dict[str, torch.Tensor],
            op: torch.func) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Apply an operation involving each pair of elements stored into two lists or dicts of Tensors.

    Params:
        a: first list or dict of Tensors.
        b: second list or dict of Tensors.
        op: operation to be applied to the elements in both the lists (it could be a lambda expression).

    Returns:
        List or dict of Tensors with the result of the operation (same size of each list-argument).
    """
    assert type(a) is type(b), f'Type of the inputs to this function should match, got {type(a)} and {type(b)} instead.'

    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys()), 'Dictionaries should have the same keys.'
        oo = {_k_a: op(a[_k_a], b[_k_a]) for _k_a in a.keys()}
    elif isinstance(a, torch.Tensor):
        oo = op(a, b)
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return oo


def _copy_inplace(a: torch.Tensor | Dict[str, torch.Tensor], b: torch.Tensor | Dict[str, torch.Tensor],
                  detach: bool = False) -> None:
    """Copies 'in-place' the values of a list or dict of Tensors (b) into another one (a).

    Params:
        a: list or dict of Tensors.
        b: another list or dict of Tensors, same sizes of the one above.
    """
    assert type(a) is type(b), f'Type of the inputs to this function should match, got {type(a)} and {type(b)} instead.'

    if detach:
        b = _detach(b)
    if isinstance(a, torch.Tensor):
        a.copy_(b)
    elif isinstance(a, dict):
        for _k_b in b.keys():
            a[_k_b].copy_(b[_k_b])
    else:
        raise Exception(f'Inputs to this function should be either lists or dicts, got {type(a)} and {type(b)}.')


def _copy(a: torch.Tensor | Dict[str, torch.Tensor], detach: bool = False) \
        -> torch.Tensor | Dict[str, torch.Tensor]:
    """Copies the values of a list or dict of Tensors into another one.

    Params:
        a: list or dict of Tensors.

    Returns:
        b: another list or dict of Tensors, same sizes of the one above.
    """
    if detach:
        a = _detach(a)
    if isinstance(a, torch.Tensor):
        b = a.clone()
    elif isinstance(a, dict):
        b = {k: v.clone() for k, v in a.items()}
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return b


def _zero_grad(tensors: torch.Tensor | Dict[str, torch.Tensor], set_to_none: bool = False) -> None:
    """Zeroes the gradient field of a list or dict of Tensors.

    Params:
        tensors: a Tensor or a list or dict of Tensors with requires_grad activated.
        set_to_none: forces all 'grad' fields to be set to None.
    """
    if isinstance(tensors, dict):
        list_ten = list(tensors.values())
    elif isinstance(tensors, torch.Tensor):
        list_ten = [tensors, ]
    else:
        raise Exception(f'Input to this function should be either list or dict, or a Tensor, got {type(tensors)}.')
    for a in list_ten:
        if a.grad is not None:
            if set_to_none:
                a.grad = None
            else:
                if a.grad.grad_fn is not None:
                    a.grad.detach_()
                else:
                    a.grad.requires_grad_(False)
                    a.grad.zero_()


def _zero(tensors: torch.Tensor | Dict[str, torch.Tensor], detach: bool = False) \
        -> torch.Tensor | Dict[str, torch.Tensor]:
    """Returns a zeroed copy of a list or dict of tensors.

    Params:
        tensors: a list or dict of Tensors.
    """
    if detach:
        tensors = _detach(tensors)
    if isinstance(tensors, torch.Tensor):
        b = torch.zeros_like(tensors)
    elif isinstance(tensors, dict):
        b = {k: torch.zeros_like(v) for k, v in tensors.items()}
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(tensors)}.')

    return b


def _zero_inplace(tensors: torch.Tensor | Dict[str, torch.Tensor], detach: bool = False) -> None:
    """Zeroes a list or dict of Tensors (inplace).

    Params:
        tensors: a list or dict of Tensors.
    """
    if detach:
        tensors = _detach(tensors)
    if isinstance(tensors, dict):
        for a in tensors.values():
            a.zero_()
    elif isinstance(tensors, torch.Tensor):
        tensors.zero_()
    else:
        raise ValueError('Unsupported type.')


def _detach(tensors: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Detaches a list or dict of Tensors (not-in-place).

    Params:
        a: a list or dict of Tensors.

    Returns:
        A list or dict of detached Tensors.
    """
    if isinstance(tensors, dict):
        oo = dict.fromkeys(tensors)
        for k, a in tensors.items():
            oo[k] = a.detach()
    elif isinstance(tensors, torch.Tensor):
        oo = tensors.detach()
    else:
        raise ValueError('Unsupported type.')
    return oo


class HL:
    def __init__(self, models: torch.nn.Module | Iterable[Dict[str, torch.nn.Module | Any]], *,
                 gamma=1., flip=-1., theta=0.1, beta=1., reset_neuron_costate=False, reset_weight_costate=False,
                 local=True):
        """
        Args:
            models (list of dict): List of parameter groups, each containing:
                - 'params': the Model or list of parameters
                - 'gamma', 'beta', 'theta', etc.: Hyperparameters for the group.
        """

        # Set defaults
        defaults = dict(params=None, gamma=gamma, flip=flip, theta=theta, beta=beta,
                        reset_neuron_costate=reset_neuron_costate, reset_weight_costate=reset_weight_costate,
                        local=local)

        # Ensure models is a list of dicts and assign the specified values
        if isinstance(models, torch.nn.Module):
            models = [{**defaults, 'params': models}]

        self.param_groups = []
        for group in models:
            assert 'params' in group, "Each parameter group must contain a 'params' key storing the model."
            self.param_groups.append({**defaults, **group})

        # Store the optimizer state for each model in a list of dicts, not to be confused with the state of the model
        self.state = [_init_state_and_costate(group['params']) for group in self.param_groups]

    @torch.no_grad()
    def step(self):
        """Perform one optimization step for all parameter groups."""

        for group, state in zip(self.param_groups, self.state):
            model = group['params']
            delta = model.delta

            # Copy the state (of the model) just to track it during the optimization and get the costate
            # the locality of these operations is handled by the model
            state['x']['xi'] = model.h
            dp_xi = _get_grad(model.h)
            _euler_step(state['p']['xi'], dp_xi, step_size=-delta * group['flip'],
                        decay=-group['flip'] * group['theta'], in_place=True)

            # Copy the weights from the network just to track it during the optimization and get the costates
            dp_w = {}
            for name, param in model.named_parameters():
                state['x']['w'][name] = param
                dp_w[name] = _get_grad(param)

            if group['local']:

                # Local HL uses the old costates to update the weights
                d_w = state['p']['w']
                _euler_step(state['x']['w'], d_w, step_size=-delta*group['beta'], decay=None, in_place=True)
                _euler_step(state['p']['w'], dp_w, step_size=-delta*group['flip'],
                            decay=-group['flip']*group['theta'], in_place=True)
            else:

                # Non-local HL updates the costates before updating the weights
                d_w = _euler_step(state['p']['w'], dp_w, step_size=-delta * group['flip'],
                                  decay=-group['flip'] * group['theta'], in_place=True)
                _euler_step(state['x']['w'], d_w, step_size=-delta * group['beta'], decay=None, in_place=True)

    def compute_hamiltonian(self, *potential_terms: torch.Tensor) -> torch.Tensor:
        """Computes the Hamiltonian for all models."""

        # The number of potential terms provided should be equal to the number of models
        assert len(potential_terms) == len(self.param_groups), f"A potential term for each model is expected."

        ham = torch.tensor(0., dtype=potential_terms[0].dtype, device=potential_terms[0].device)
        for group, state, potential_term in zip(self.param_groups, self.state, potential_terms):
            model = group['params']
            ham += group['gamma'] * potential_term + torch.dot(model.dh.view(-1), state['p']['xi'].view(-1)).real
        return ham

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zeroes the gradients and resets co-states if needed."""

        for group, state in zip(self.param_groups, self.state):
            model = group['params']
            _zero_grad(model.h, set_to_none)
            for param in model.parameters():
                _zero_grad(param, set_to_none)

            # Eventually reset costates
            if group['reset_neuron_costate']:
                _zero_inplace(state['p']['xi'], detach=True)
            if group['reset_weight_costate']:
                _zero_inplace(state['p']['w'], detach=True)
