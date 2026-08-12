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
import json
import psutil
import hashlib
import platform
import datetime
import requests
import ipaddress
from datetime import timezone
from unaiverse.utils.logger import log


class NodeProfile:
    """
    Profile information for a node.
    """

    def __init__(self,
                 static: dict,
                 dynamic: dict,
                 cv: list) -> None:
        """Initializes a NodeProfile from static, dynamic, and CV data.

        Args:
            static: Dictionary of static profile fields (e.g. node_id, node_type, name, nickname).
            dynamic: Dictionary of dynamic profile fields (e.g. os, memory, peer_id).  Only
                keys that are expected by the internal template are copied in.
            cv: List of CV entry dictionaries; entries are sorted by ``last_edit_utc`` before storage.

        Raises:
            ValueError: If ``static`` is empty or any required static key is missing.
        """

        # Checking provided data
        if not static:
            raise ValueError("Missing static profile data")

        # Forcing key order (important! otherwise the hash operation will not be consistent with the one on the server)
        cv = [{k: _cv[k] for k in sorted(_cv)} for _cv in sorted(cv, key=lambda x: x['last_edit_utc'])]

        self._profile_data: dict = \
            {
                'static': {
                    'node_id': None,
                    'node_type': None,
                    'node_name': None,
                    'node_description': None,
                    'created_utc': None,
                    'name': None,
                    'surname': None,
                    'title': None,
                    'nickname': None,  # The owner's public identity
                    'organization': None,  # Legacy: the server no longer sends it
                    'email': None,  # Legacy: the server no longer sends it
                    'max_nr_connections': None,
                    'allowed_node_ids': None,
                    'world_masters_node_ids': None,
                    'certified': None,
                    'inspector_node_id': None,
                    'location_method': None,
                    'location': None
                },
                'dynamic': {
                    'os': None,
                    'cpu_cores': None,
                    'logical_cpus': None,
                    'memory_gb': None,
                    'memory_avail': None,
                    'memory_used': None,
                    'timestamp': None,
                    'public_ip_address': None,
                    'guessed_location': None,
                    'peer_id': None,
                    'peer_addresses': None,
                    'private_peer_id': None,
                    'private_peer_addresses': None,
                    'is_relay': None,  # True when this node is a world offering a reachable relay to its members
                    'proc_inputs': None,
                    'proc_outputs': None,
                    'streams': None,
                    'connections': {
                        'public_agents': None,  # List of dict
                        'world_agents': None,  # List of dict
                        'world_masters': None,  # List of dict
                        'world_peer_id': None,  # Str
                        'role': None  # Str
                    },
                    'world_summary': {
                        "world_title": None,
                        "world_agents": None,
                        "world_masters": None,
                        "world_agents_count": None,
                        "world_masters_count": None,
                        "total_agents": None,
                        "agent_badges_count": None,
                        "agent_badges": None,
                        "streams_count": None
                    },
                    "hidden": None
                },
                'cv': cv
            }

        # Backward compatibility
        if "location" not in static:
            static['location'] = {}
        if "location_method" not in static:
            static['location_method'] = "manual"

        # Only the keys the SDK cannot run without are required: every other template key
        # defaults to None, so a field the server stops sending degrades instead of
        # crashing the boot
        for k in ('node_id', 'node_type', 'node_name'):
            if k not in static:
                raise ValueError("Missing required static profile info: " + str(k))

        # Filling static profile info (there might be more information that the one shown above)
        for k, v in static.items():
            self._profile_data['static'][k] = v

        # Including the provided dynamic info, only considering the expected keys
        # (the provided "dynamic" argument will contain all or just a sub-portion of the expected keys)
        for k, v in dynamic.items():
            if k == 'connections' and v is not None and isinstance(v, dict):
                for kk, vv in v.items():
                    if (kk in self._profile_data['dynamic']['connections'] and
                            self._profile_data['dynamic']['connections'][kk] is None):
                        self._profile_data['dynamic']['connections'][kk] = vv
            elif k == 'world_summary' and v is not None and isinstance(v, dict):
                for kk, vv in v.items():
                    if (kk in self._profile_data['dynamic']['world_summary'] and
                            self._profile_data['dynamic']['world_summary'][kk] is None):
                        self._profile_data['dynamic']['world_summary'][kk] = vv
            elif k in self._profile_data['dynamic'] and self._profile_data['dynamic'][k] is None:
                self._profile_data['dynamic'][k] = v
            elif k.startswith('tmp_'):
                self._profile_data['dynamic'][k] = v

        # Internally required attributes
        self._profile_last_updated = None  # Will be set by calling _fill_missing_specs or check_and_update_specs
        self._geolocation_cache = {}  # Will be needed to avoid too many IP-related lookups

        # Filling the missing information (machine-level information, specs) that can be automatically extracted
        self._fill_missing_specs()

        # Flag
        self._connections_updated = False

    def update_cv(self, new_cv: list[dict]) -> None:
        """Replaces the stored CV data with a new list of CV entries.

        Args:
            new_cv: The new CV data as a list of dictionaries.
        """
        self._profile_data['cv'] = new_cv

    @classmethod
    def from_dict(cls, combined_data: dict) -> 'NodeProfile':
        """Creates a NodeProfile instance from a combined profile dictionary.

        Args:
            combined_data: A dictionary representing the node profile, typically loaded from
                JSON or received over the network.  Expected to contain a ``"static"`` key
                (with at least ``"node_id"``), a ``"dynamic"`` key, and a ``"cv"`` key
                (list of dicts).

        Returns:
            A new NodeProfile instance populated from the dictionary.

        Raises:
            ValueError: If ``"node_id"`` is absent from the ``"static"`` sub-dictionary.
            TypeError: If the ``"cv"`` data is present but not a list.
        """

        # Ensure essential 'node_id' is present
        static_combined_data = combined_data.get('static')
        if static_combined_data is None:
            raise ValueError("Input dictionary must contain a 'static' section.")
        static_combined_data: dict
        node_id = static_combined_data.get('node_id', None)
        if not node_id:
            raise ValueError("Input dictionary must contain a 'node_id'.")

        profile_instance = cls(
            static=combined_data['static'],
            dynamic=combined_data['dynamic'],
            cv=combined_data['cv']
        )

        return profile_instance

    # Get operating system information
    @staticmethod
    def _get_os_spec() -> str:
        """Extracts the operating system platform string.

        Returns:
            A human-readable string identifying the current OS and version.
        """
        return platform.platform()

    # Get cpu information
    @staticmethod
    def _get_cpu_info() -> dict:
        """Extracts CPU core count information.

        Returns:
            A dictionary with keys ``"physical_cores"`` and ``"logical_cores"`` (both ``int``
            or ``None`` on error).
        """
        try:
            return {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True)
            }
        except Exception as e:
            log.error(f"Error getting CPU info: {e}")
            return {'physical_cores': None, 'logical_cores': None}

    # Get memory information
    @staticmethod
    def _get_memory_info() -> dict:
        """Extracts system memory statistics in gigabytes.

        Returns:
            A dictionary with keys ``"total"``, ``"available"``, and ``"used"`` (all ``float``).
            Returns zeros on error.
        """
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            return {
                'total': float(total_gb),
                'available': float(available_gb),
                'used': float(used_gb)
            }
        except Exception as e:
            log.error(f"Error getting memory info: {e}")
            return {'total': 0.0, 'available': 0.0, 'used': 0.0}

    # Get public ip address
    @staticmethod
    def _get_public_ip_address() -> str:
        """Attempts to retrieve the public IP address using an external web service.

        Tries several services in order and returns the first valid IPv4/IPv6 address found.
        If all services fail, returns the string ``'Public IP not available.'``.

        Returns:
            The public IP address as a string, or ``'Public IP not available.'`` if all
            lookup attempts fail.
        """

        # List of reliable services that return the public IP as plain text
        services = [
            "https://api.ipify.org",
            "https://icanhazip.com",
            "https://ident.me",
            "https://checkip.amazonaws.com",
        ]

        # Print("Attempting to retrieve public IP address...")
        for url in services:
            try:

                # Make a GET request to the service URL with a retry_timeout
                response = requests.get(url, timeout=5)

                # Raise an HTTPError for bad responses (4xx or 5xx status codes)
                response.raise_for_status()

                # Get the response text, which should be the IP address, and strip any whitespace
                public_ip = response.text.strip()

                # Basic validation - check if the result looks like a valid IP address
                try:
                    ipaddress.ip_address(public_ip)  # This checks if it's a valid IPv4 or IPv6 address

                    return public_ip  # Return the first valid IP found

                except ValueError:

                    # If ipaddress.ip_address raises ValueError, it's not a valid format
                    continue  # Try the next service if validation fails

            except requests.exceptions.RequestException:

                # Catch any request-related errors (e.g., network issues, retry_timeout, bad status)
                continue  # Try the next service on error

            except Exception:

                # Catch any other unexpected errors
                continue  # Try the next service on error

        return 'Public IP not available.'

    # Get guessed location based on IP address
    def _get_geolocation_from_ip(self, ip_address: str) -> dict:
        """Retrieves geolocation data for the given IP address via the ip-api.com service.

        Results are cached in ``self._geolocation_cache`` to avoid repeated API calls.
        Private, loopback, and unspecified addresses are handled locally without a network call.

        Args:
            ip_address: The IPv4 or IPv6 address string to geolocate.

        Returns:
            A dictionary with geolocation fields (``"country"``, ``"city"``, ``"latitude"``,
            ``"longitude"``, etc.) on success, or a dictionary with an ``"error"`` key on
            failure or for non-routable addresses.
        """

        # Added a check for local/private IPs to avoid unnecessary API calls
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_unspecified:
                return {"message": "Private, loopback, or unspecified IP address. Geolocation not applicable."}
        except ValueError:
            return {"error": f"Invalid IP address format: {ip_address}"}

        # Added a simple cache to avoid repeated API calls for the same IP
        if hasattr(self, '_geolocation_cache') and ip_address in self._geolocation_cache:

            # Print(f"Using cached geolocation for {ip_address}") # Optional: for debugging
            return self._geolocation_cache[ip_address]

        try:
            url = f"http://ip-api.com/json/{ip_address}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                geo_data = {
                    "country": data.get("country"),
                    "countryCode": data.get("countryCode"),
                    "region": data.get("region"),
                    "regionName": data.get("regionName"),
                    "city": data.get("city"),
                    "zip": data.get("zip"),
                    "latitude": data.get("lat"),
                    "longitude": data.get("lon"),
                    "timezone": data.get("timezone"),
                    "isp": data.get("isp")
                }

                # Cache the result
                if not hasattr(self, '_geolocation_cache'):
                    self._geolocation_cache = {}
                self._geolocation_cache[ip_address] = geo_data
                return geo_data
            else:
                error_data = {"error": data.get("message", "Geolocation lookup failed.")}

                # Cache the error result too
                if not hasattr(self, '_geolocation_cache'):
                    self._geolocation_cache = {}
                self._geolocation_cache[ip_address] = error_data
                return error_data

        except requests.exceptions.RequestException as e:
            error_data = {"error": f"Request failed: {e}"}
            if not hasattr(self, '_geolocation_cache'):
                self._geolocation_cache = {}
            self._geolocation_cache[ip_address] = error_data
            return error_data

        except json.JSONDecodeError:
            error_data = {"error": "Failed to decode JSON response from geolocation API"}
            if not hasattr(self, '_geolocation_cache'):
                self._geolocation_cache = {}
            self._geolocation_cache[ip_address] = error_data
            return error_data

        except Exception as e:
            error_data = {"error": f"An unexpected error occurred during geolocation lookup: {e}"}
            if not hasattr(self, '_geolocation_cache'):
                self._geolocation_cache = {}
            self._geolocation_cache[ip_address] = error_data
            return error_data

    # This is the function that collects all the information for the 'node_specification'
    def _get_current_specs(self) -> dict:
        """Gathers current system specifications (OS, CPU, memory, IP, and location).

        Returns:
            A dictionary suitable for merging into the dynamic profile, containing
            ``"timestamp"``, ``"os"``, ``"cpu_cores"``, ``"logical_cpus"``, ``"memory_gb"``,
            ``"memory_avail"``, ``"memory_used"``, ``"public_ip_address"``, and
            ``"guessed_location"``.
        """
        cpu_info = self._get_cpu_info()
        memory_info = self._get_memory_info()

        location = self._profile_data['static'].get('location', {})
        location_method = self._profile_data['static'].get('location_method', "manual")
        return {
            'timestamp': datetime.datetime.now(timezone.utc).isoformat(),
            'os': self._get_os_spec(),
            'cpu_cores': cpu_info.get('physical_cores'),
            'logical_cpus': cpu_info.get('logical_cores'),
            'memory_gb': memory_info.get('total'),
            'memory_avail': memory_info.get('available'),
            'memory_used': memory_info.get('used'),
            'public_ip_address': self._get_public_ip_address(),
            'guessed_location': self._get_geolocation_from_ip(self._get_public_ip_address())
            if location_method != "manual" else location
        }

    def _fill_missing_specs(self) -> None:
        """Fills any ``None`` fields in the dynamic profile with current system specs.

        Only calls ``_get_current_specs`` if at least one dynamic field is still ``None``.
        Also updates ``_profile_last_updated``.
        """
        dynamic_profile = self.get_dynamic_profile()
        current_specs = None
        for k in dynamic_profile.keys():
            if dynamic_profile[k] is None:
                if current_specs is None:
                    current_specs = self._get_current_specs()
                if k in current_specs:
                    dynamic_profile[k] = current_specs[k]

        self._profile_last_updated = datetime.datetime.now(timezone.utc)  # Mark profile as checked/updated

    def check_and_update_specs(self, update_only: bool = True) -> bool:
        """Checks current system specs and updates the dynamic profile accordingly.

        Args:
            update_only: If True, unconditionally merges current specs into the dynamic
                profile without change detection.  If False, compares each spec field
                against the saved value and only merges when a change is detected
                (default: True).

        Returns:
            True if any spec field changed (only meaningful when ``update_only=False``),
            False otherwise.
        """

        current_specs = self._get_current_specs()
        specs_changed = False

        if update_only:
            self._profile_data['dynamic'] |= current_specs
        else:
            saved_specs = self._profile_data['dynamic'].copy()
            change_details = []

            # Compare current specs with saved specs (ignore timestamp for comparison)
            keys_to_compare = current_specs.keys()

            for key in keys_to_compare:
                if key == 'timestamp':
                    continue

                saved_value = saved_specs.get(key)
                current_value = current_specs.get(key)

                # Handle float comparison with tolerance
                if isinstance(saved_value, float) and isinstance(current_value, float):
                    if abs(current_value - saved_value) > 1e-6:  # Tolerance for float changes
                        change_details.append(f"{key}: from {saved_value:.2f} to {current_value:.2f}")
                        specs_changed = True

                elif saved_value != current_value:
                    change_details.append(f"{key}: from {saved_value} to {current_value}")
                    specs_changed = True

            # Comparing total resources (OS, CPU, total RAM/Disk) is more typical for 'specification' changes.
            if specs_changed:
                # Update the specification in the profile data with the new current specs
                self._profile_data['dynamic'] |= current_specs
                change_summary = ", ".join(change_details)
                log.print(f"Specs changed for '{self._profile_data['static']['node_id']}': {change_summary}")

        self._profile_last_updated = datetime.datetime.now(timezone.utc)  # Mark profile as checked/updated

        return specs_changed

    # Get profile data as dict: cv, dynamic_profile, static_profile
    # TODO(review): these getters (and get_dynamic_profile/get_cv/get_all_profile below) return the internal mutable
    # dict/list directly instead of a copy, so callers can mutate the profile's private state (encapsulation leak);
    # see tests/test_profile.py::test_get_all_profile_structure
    def get_static_profile(self) -> dict:
        """Returns the static portion of the profile data.

        Returns:
            A dictionary containing static profile fields such as node_id, node_type,
            and user identity information.
        """
        return self._profile_data['static']

    def get_dynamic_profile(self) -> dict:
        """Returns the dynamic portion of the profile data.

        Returns:
            A dictionary containing dynamic profile fields such as OS, CPU, memory,
            IP address, peer information, and connection state.
        """
        return self._profile_data['dynamic']

    def get_cv(self) -> list:
        """Returns the CV data associated with this node profile.

        Returns:
            A list of CV entry dictionaries, sorted by ``last_edit_utc``.
        """
        return self._profile_data['cv']

    def get_all_profile(self) -> dict:
        """Returns the complete profile data dictionary.

        Returns:
            A dictionary with keys ``"static"``, ``"dynamic"``, and ``"cv"``.
        """
        return self._profile_data

    def mark_change_in_connections(self) -> None:
        """Flags that a connection change has occurred since the last reset."""
        self._connections_updated = True

    def unmark_change_in_connections(self) -> None:
        """Clears the connection-change flag."""
        self._connections_updated = False

    def connections_changed(self) -> bool:
        """Returns whether a connection change has been recorded since the last reset.

        Returns:
            True if connections have changed, False otherwise.
        """
        return self._connections_updated

    def verify_cv_hash(self, cv_hash: str) -> tuple[bool, tuple[str, str]]:
        """Verifies a CV hash against the hash computed from the stored CV data.

        Args:
            cv_hash: The hash string to verify.

        Returns:
            A tuple ``(match, (provided_hash, computed_hash))`` where ``match`` is True if
            the hashes are equal, and the second element contains both hash strings for
            diagnostic purposes.
        """
        cv = self._profile_data['cv']  # Should we sort keys?
        computed_hash = hashlib.blake2b(json.dumps(cv).encode("utf-8"),
                                        digest_size=16).hexdigest()
        return cv_hash == computed_hash, (cv_hash, computed_hash)
