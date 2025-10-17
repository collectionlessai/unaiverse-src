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
import ntplib
import bisect
import socket
from ntplib import NTPException
from datetime import datetime, timezone


class Clock:
    """
    A class for managing time cycles and converting between timestamp and cycle indices.

    This class interacts with an NTP server to synchronize time and supports operations
    to track cycles, manage timestamps, and calculate the time differences between cycles.
    """

    def __init__(self, min_delta: float = -1):
        """Initialize a Clock instance.

        Args:
            min_delta (float): Minimum time (in seconds) between consecutive cycles.
                                If less than or equal to zero, the cycles will be real-time-based.
        """
        self.min_delta = min_delta  # Min-time passed between consecutive cycles (seconds) - if <=0, it is real-time
        self.cycle = -1  # Internal index, not shared outside (the value -1 is only used at creation/reset time)
        self.__servers = [
            'pool.ntp.org',
            'north-america.pool.ntp.org'
            'asia.pool.ntp.org',
            'europe.pool.ntp.org',
        ]
        self.__global_initial_t = self.__get_time_from_server()  # Real-time, wall-clock
        if self.__global_initial_t == -1.:
            raise ValueError("Unable to get the initial time (for synchronization purposes) from the NTP servers")
        self.__local_initial_t = datetime.now(timezone.utc).timestamp()  # Corresponding local time
        self.__timestamps = []  # List to store timestamps for cycles
        self.__time2cycle_cache = 0  # Cached cycle value for optimization

    def __get_time_from_server(self) -> float:
        """Get the current time from an NTP server.

        Returns:
            float: The time returned by the NTP server, converted to a timestamp.
        """
        c = ntplib.NTPClient()
        response = None
        for i in range(0, 10):
            try:
                server = self.__servers[i % len(self.__servers)]
                response = c.request(server, version=3)
                break
            except (NTPException, socket.gaierror):
                continue
        if response is not None:
            return datetime.fromtimestamp(response.tx_time, timezone.utc).timestamp()
        else:
            return -1.

    def __add_timestamp(self, timestamp: float):
        """Add a timestamp to the list of timestamps for the clock cycles.

        Args:
            timestamp (float): The timestamp to be added to the list.

        Raises:
            ValueError: If the provided timestamp is not more recent than the last one.
        """
        if len(self.__timestamps) == 0 or self.__timestamps[-1] < timestamp:
            self.__timestamps.append(timestamp)
        else:
            raise ValueError("Cannot add a timestamp that is NOT more recent than the already added ones")

    def time2cycle(self, timestamp: float, delta: float | None = None) -> int:
        """Convert a given timestamp to the corresponding cycle index.

        Args:
            timestamp (float): The timestamp to convert.
            delta (float | None): The optional delta value for converting time to cycles.

        Returns:
            int: The cycle index corresponding to the given timestamp.
        """
        if delta is not None and delta > 0:
            passed = self.get_time() - timestamp  # Precision: microseconds
            return self.cycle - int(passed * delta)
        else:
            self.__time2cycle_cache = Clock.__search(self.__timestamps, timestamp, self.__time2cycle_cache)
            return self.__time2cycle_cache

    def cycle2time(self, cycle: int, delta: float | None = None) -> float:
        """Convert a cycle index to the corresponding timestamp.

        Args:
            cycle (int): The cycle index to convert.
            delta (float | None): The optional delta value for converting cycles to time.

        Returns:
            float: The timestamp corresponding to the given cycle index.
        """
        if delta is not None and delta > 0:
            return cycle * delta
        else:
            return self.__timestamps[cycle] if cycle >= 0 else -1.

    def get_time(self, passed: bool = False) -> float:
        """Get the current time based on the NTP server synchronization.

        Returns:
            float: The current synchronized time (in seconds since the Unix epoch).
        """
        passed_since_beginning = datetime.now(timezone.utc).timestamp() - self.__local_initial_t
        return self.__global_initial_t + passed_since_beginning if not passed else passed_since_beginning

    def get_time_as_string(self) -> str:
        """Get the current time as a string (ISO format).

        Returns:
            str: A string representation of the current time (ISO format, UTC).
        """
        dt_object = datetime.fromtimestamp(self.get_time(), tz=timezone.utc)
        return dt_object.isoformat(timespec='milliseconds')

    def get_cycle(self):
        """Get the current cycle index.

        Returns:
            int: The current cycle index.
        """
        return self.cycle

    def get_cycle_time(self):
        """Get the timestamp corresponding to the current cycle.

        Returns:
            float: The timestamp corresponding to the current cycle index.
        """
        return self.cycle2time(self.cycle)

    def next_cycle(self) -> bool:
        """Move to the next cycle if the minimum delta time has passed or if cycles are not constrained.

        Returns:
            bool: True if the cycle was successfully moved to the next one, False otherwise.
        """
        if self.cycle >= 0 and (self.min_delta > 0 and len(self.__timestamps) > 0 and
                                (self.get_time() - self.__timestamps[-1]) < self.min_delta):
            return False
        else:
            self.cycle += 1  # Increment the cycle index
            self.__add_timestamp(self.get_time())
            return True

    @staticmethod
    def __search(_list, _target, _last_pos):
        """Search for a target value in the list of timestamps and return the index of the corresponding cycle.

        Args:
            _list (list): The list of timestamps.
            _target (float): The target timestamp to search for.
            _last_pos (int): The last search position, used for optimization.

        Returns:
            int: The index of the found timestamp in the list, or -1 if not found.
        """
        if len(_list) > 0 and _target > _list[-1]:
            return len(_list)
        if len(_list) > _last_pos and _list[_last_pos] == _target:
            return _last_pos
        elif len(_list) > (_last_pos + 1) and _list[_last_pos + 1] == _target:
            return _last_pos + 1
        elif len(_list) == 0:
            return -1
        else:
            ret = bisect.bisect_left(_list, _target)
            if _list[ret] == _target:
                return ret
            else:
                return -1
