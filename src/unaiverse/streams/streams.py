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
import math
import torch
from unaiverse.streams import BufferedDataStream, DataProps


class AllHotLabelStream(BufferedDataStream):
    """
    A buffered stream that simply repeat a single-element tensor valued "ones" (float), associated to some text labels
    """

    def __init__(self, labels: list[str],
                 device: torch.device = torch.device('cpu')):
        super().__init__(props=DataProps(name=AllHotLabelStream.__name__,
                                         data_type="tensor",
                                         data_desc="dummy stream",
                                         tensor_shape=(1, len(labels)),
                                         tensor_labels=labels,
                                         tensor_dtype=str(torch.float),
                                         tensor_labeling_rule="geq0.5"),
                         is_static=True)

        self.set(torch.ones((1, len(labels)), dtype=torch.float32, device=device))
        self.restart()


class Random(BufferedDataStream):

    def __init__(self, std: float, shape: tuple[int] | None = (1,),
                 device: torch.device = torch.device('cpu')):
        super().__init__(props=DataProps(name="rand",
                                         data_type="tensor",
                                         data_desc="stream of random numbers",
                                         tensor_shape=shape,
                                         tensor_dtype=str(torch.float)))
        self.std = std
        self.device = device
        self.restart()

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        y = self.std * torch.rand(self.props.tensor_shape, device=self.device)
        return self.props.adapt_tensor_to_tensor_labels(y), self.clock.get_cycle() - self.first_cycle


class Sin(BufferedDataStream):

    def __init__(self, freq: float, phase: float, delta: float,
                 device: torch.device = torch.device('cpu')):
        super().__init__(props=DataProps(name="sin",
                                         data_type="tensor",
                                         data_desc="stream of samples from the sin function",
                                         tensor_shape=(1, 1),
                                         tensor_dtype=str(torch.float)))
        self.freq = freq
        self.phase = phase
        self.period = 1. / self.freq
        self.delta = delta
        self.device = device
        self.restart()

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        t = idx * self.delta + self.phase * self.period
        y = torch.sin(torch.tensor([[2. * math.pi * self.freq * t]], device=self.device))
        return self.props.adapt_tensor_to_tensor_labels(y), self.clock.get_cycle() - self.first_cycle


class Square(BufferedDataStream):

    def __init__(self, freq: float, ampl: float, phase: float, delta: float,
                 device: torch.device = torch.device('cpu')):
        super().__init__(props=DataProps(name="square",
                                         data_type="tensor",
                                         data_desc="stream of samples from the square function",
                                         tensor_shape=(1, 1),
                                         tensor_dtype=str(torch.float)))
        self.freq = freq
        self.ampl = ampl
        self.phase = phase
        self.period = 1. / self.freq
        self.delta = delta
        self.device = device
        self.restart()

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        t = idx * self.delta + self.phase * self.period
        y = self.ampl * torch.tensor([[(-1.) ** (math.floor(2. * self.freq * t))]], device=self.device)
        return self.props.adapt_tensor_to_tensor_labels(y), self.clock.get_cycle() - self.first_cycle


class CombSin(BufferedDataStream):

    def __init__(self, f_cap: float | list, c_cap: float | list, order: int, delta: float,
                 device: torch.device = torch.device('cpu')):
        super().__init__(props=DataProps(name="combsin",
                                         data_type="tensor",
                                         data_desc="stream of samples from combined sin functions",
                                         tensor_shape=(1, 1),
                                         tensor_dtype=str(torch.float)))
        if isinstance(f_cap, float):
            self.freqs = f_cap * torch.rand(order)
        elif isinstance(f_cap, list):
            self.freqs = torch.tensor(f_cap)
        else:
            raise Exception(f"expected float or list for f_cap, not {type(f_cap)}")
        self.phases = torch.zeros_like(self.freqs)
        if isinstance(c_cap, float):
            self.coeffs = c_cap * (2 * torch.rand(order) - 1)
        elif isinstance(c_cap, list):
            self.coeffs = torch.tensor(c_cap)
        else:
            raise Exception(f"expected float or list for c_cap, not {type(c_cap)}")

        # Check all the dimensions
        assert len(self.coeffs) == len(self.freqs), \
            (f"specify the same number of coefficients and frequencies (got {len(self.coeffs)} "
             f"and {len(self.freqs)} respectively).")

        self.delta = delta
        self.device = device
        self.restart()

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        t = idx * self.delta
        y = torch.sum(self.coeffs * torch.sin(2 * math.pi * self.freqs * t + self.phases)).view(1, 1)
        return self.props.adapt_tensor_to_tensor_labels(y), self.clock.get_cycle() - self.first_cycle


class SmoothHFHA(CombSin):
    FEATURES = ['3sin', 'hf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')):
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.8, 0.16, 0.16]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoHfHa")


class SmoothHFLA(CombSin):
    FEATURES = ['3sin', 'hf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')):
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.4, 0.08, 0.08]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoHfLa")


class SmoothLFLA(CombSin):
    FEATURES = ['3sin', 'lf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')):
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.08, 0.08, 0.4]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoLfLa")


class SmoothLFHA(CombSin):
    FEATURES = ['3sin', 'lf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')):
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.16, 0.16, 0.8]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoLfHa")


class SquareHFHA(Square):
    FEATURES = ['square', 'hf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__(freq=0.06, phase=0.5, ampl=1.0, delta=0.1, device=device)
        self.props.set_name("squHfHa")


class SquareHFLA(Square):
    FEATURES = ['square', 'hf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__(freq=0.06, phase=0.5, ampl=0.5, delta=0.1, device=device)
        self.props.set_name("squHfLa")


class SquareLFHA(Square):
    FEATURES = ['square', 'lf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__(freq=0.03, phase=0.5, ampl=1.0, delta=0.1, device=device)
        self.props.set_name("squLfHa")


class SquareLFLA(Square):
    FEATURES = ['square', 'lf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__(freq=0.03, phase=0.5, ampl=0.5, delta=0.1, device=device)
        self.props.set_name("squLfLa")
