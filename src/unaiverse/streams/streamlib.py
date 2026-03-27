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
from unaiverse.clock import clock
from unaiverse.streams.streams import BufferedStream, DataProps


class AllHotLabelStream(BufferedStream):
    """
    A buffered stream that simply repeat a single-element tensor valued "ones" (float), associated to some text labels
    """

    def __init__(self, labels: list[str],
                 device: torch.device = torch.device('cpu')) -> None:
        """Initialize an AllHotLabelStream with the given label names.

        Args:
            labels: List of label strings to associate with the all-ones tensor.
            device: PyTorch device to place the tensor on (default: CPU).
        """
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


class Random(BufferedStream):
    """A buffered stream that yields random tensors drawn from a zero-centred uniform distribution."""

    def __init__(self, std: float, shape: tuple[int, ...] | None = (1,),
                 device: torch.device = torch.device('cpu')) -> None:
        """Initialize a Random stream.

        Args:
            std: Scale factor applied to the uniformly-sampled values (standard-deviation-like).
            shape: Shape of the output tensor (default: ``(1, )``).
            device: PyTorch device to place the tensor on (default: CPU).
        """
        super().__init__(props=DataProps(name="rand",
                                         data_type="tensor",
                                         data_desc="stream of random numbers",
                                         tensor_shape=shape,
                                         tensor_dtype=str(torch.float)))
        self.std = std
        self.device = device
        self.restart()

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | None, int]:
        """Generate and return a random tensor for the given index.

        Args:
            idx_and_uuid: Tuple of (index, UUID); the index is unused for on-the-fly generation.

        Returns:
            A tuple of the generated tensor and the current clock cycle offset.
        """
        y = self.std * torch.rand(self.props.tensor_shape, device=self.device)
        return self.props.adapt_tensor_to_tensor_labels(y), clock.get_cycle() - self.first_cycle_by_uuid[None]


class Sin(BufferedStream):
    """A buffered stream that generates samples from a sine function."""

    def __init__(self, freq: float, phase: float, delta: float,
                 device: torch.device = torch.device('cpu')) -> None:
        """Initialize a Sin stream.

        Args:
            freq: Frequency of the sine wave in Hz.
            phase: Phase offset in units of full periods (0 = no offset, 1 = full period offset).
            delta: Time step between consecutive samples in seconds.
            device: PyTorch device to place the tensor on (default: CPU).
        """
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

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | None, int]:
        """Compute and return a sine sample for the given index.

        Args:
            idx_and_uuid: Tuple of (index, UUID); UUID is unused.

        Returns:
            A tuple of the sine tensor sample and the current clock cycle offset.
        """
        idx, _ = idx_and_uuid
        t = idx * self.delta + self.phase * self.period
        y = torch.sin(torch.tensor([[2. * math.pi * self.freq * t]], device=self.device))
        return self.props.adapt_tensor_to_tensor_labels(y), clock.get_cycle() - self.first_cycle_by_uuid[None]


class Square(BufferedStream):
    """A buffered stream that generates samples from a square wave function."""

    def __init__(self, freq: float, ampl: float, phase: float, delta: float,
                 device: torch.device = torch.device('cpu')) -> None:
        """Initialize a Square wave stream.

        Args:
            freq: Frequency of the square wave in Hz.
            ampl: Amplitude of the square wave.
            phase: Phase offset in units of full periods.
            delta: Time step between consecutive samples in seconds.
            device: PyTorch device to place the tensor on (default: CPU).
        """
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

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | None, int]:
        """Compute and return a square wave sample for the given index.

        Args:
            idx_and_uuid: Tuple of (index, UUID); UUID is unused.

        Returns:
            A tuple of the square wave tensor sample and the current clock cycle offset.
        """
        idx, _ = idx_and_uuid
        t = idx * self.delta + self.phase * self.period
        y = self.ampl * torch.tensor([[(-1.) ** (math.floor(2. * self.freq * t))]], device=self.device)
        return self.props.adapt_tensor_to_tensor_labels(y), clock.get_cycle() - self.first_cycle_by_uuid[None]


class CombSin(BufferedStream):
    """A buffered stream that generates samples from a sum of sine functions with random or prescribed frequencies."""

    def __init__(self, f_cap: float | list, c_cap: float | list, order: int, delta: float,
                 device: torch.device = torch.device('cpu')) -> None:
        """Initialize a CombSin stream.

        Args:
            f_cap: Maximum frequency cap (float) for random sampling, or an explicit list of frequencies.
            c_cap: Amplitude cap (float) for random coefficient sampling, or an explicit list of coefficients.
            order: Number of sine components to combine.
            delta: Time step between consecutive samples in seconds.
            device: PyTorch device to place the tensor on (default: CPU).
        """
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

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | None, int]:
        """Compute and return a combined-sine sample for the given index.

        Args:
            idx_and_uuid: Tuple of (index, UUID); UUID is unused.

        Returns:
            A tuple of the combined-sine tensor sample and the current clock cycle offset.
        """
        idx, _ = idx_and_uuid
        t = idx * self.delta
        y = torch.sum(self.coeffs * torch.sin(2 * math.pi * self.freqs * t + self.phases)).view(1, 1)
        return self.props.adapt_tensor_to_tensor_labels(y), clock.get_cycle() - self.first_cycle_by_uuid[None]


class SmoothHFHA(CombSin):
    """Preset CombSin stream: smooth signal with high frequency and high amplitude."""

    FEATURES = ['3sin', 'hf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SmoothHFHA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.8, 0.16, 0.16]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoHfHa")


class SmoothHFLA(CombSin):
    """Preset CombSin stream: smooth signal with high frequency and low amplitude."""

    FEATURES = ['3sin', 'hf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SmoothHFLA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.4, 0.08, 0.08]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoHfLa")


class SmoothLFLA(CombSin):
    """Preset CombSin stream: smooth signal with low frequency and low amplitude."""

    FEATURES = ['3sin', 'lf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SmoothLFLA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.08, 0.08, 0.4]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoLfLa")


class SmoothLFHA(CombSin):
    """Preset CombSin stream: smooth signal with low frequency and high amplitude."""

    FEATURES = ['3sin', 'lf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SmoothLFHA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        freqs = [0.11, 0.07, 0.05]
        coeffs = [0.16, 0.16, 0.8]
        super().__init__(f_cap=freqs, c_cap=coeffs, order=3, delta=0.1, device=device)
        self.props.set_name("smoLfHa")


class SquareHFHA(Square):
    """Preset Square wave stream: high frequency and high amplitude."""

    FEATURES = ['square', 'hf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SquareHFHA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        super().__init__(freq=0.06, phase=0.5, ampl=1.0, delta=0.1, device=device)
        self.props.set_name("squHfHa")


class SquareHFLA(Square):
    """Preset Square wave stream: high frequency and low amplitude."""

    FEATURES = ['square', 'hf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SquareHFLA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        super().__init__(freq=0.06, phase=0.5, ampl=0.5, delta=0.1, device=device)
        self.props.set_name("squHfLa")


class SquareLFHA(Square):
    """Preset Square wave stream: low frequency and high amplitude."""

    FEATURES = ['square', 'lf', 'ha']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SquareLFHA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        super().__init__(freq=0.03, phase=0.5, ampl=1.0, delta=0.1, device=device)
        self.props.set_name("squLfHa")


class SquareLFLA(Square):
    """Preset Square wave stream: low frequency and low amplitude."""

    FEATURES = ['square', 'lf', 'la']

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """Initialize a SquareLFLA stream.

        Args:
            device: PyTorch device to place the tensor on (default: CPU).
        """
        super().__init__(freq=0.03, phase=0.5, ampl=0.5, delta=0.1, device=device)
        self.props.set_name("squLfLa")
