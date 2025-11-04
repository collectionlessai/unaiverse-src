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


class NodeProfile:
    """
    Profile information for a node.
    """

    def __init__(self,
                 static: dict,
                 dynamic: dict,
                 cv: dict):

        # Checking provided data
        if not static:
            raise ValueError("Missing static profile data")

        # Forcing key order (important! otherwise the hash operation will not be consistent with the one on the server)
        cv = [{k: _cv[k] for k in sorted(_cv)} for _cv in cv]

        self._profile_data = \
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
                    'organization': None,
                    'email': None,
                    'max_nr_connections': None,
                    'allowed_node_ids': None,
                    'world_masters_node_ids': None,
                    'certified': None,
                    'inspector_node_id': None
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
                    "world_roles_fsm": None,  # Dict of FSMs for world roles
                    "world_stats_dynamic": None,
                    "hidden": None
                },
                'cv': cv
            }

        # Checking the presence of basic static profile info
        for k in self._profile_data['static'].keys():
            if (k not in static and k != "certified" and
                    k != "allowed_node_ids" and k != "world_masters_node_ids" and k != "inspector_node_id"):  # Patch
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

    def update_cv(self, new_cv):
        self._profile_data['cv'] = new_cv

    @classmethod
    def from_dict(cls, combined_data: dict) -> 'NodeProfile':
        """Factory method to create a NodeProfile instance from a dictionary
        containing combined profile data (static, specs, and CV list of dicts).

        Args:
            combined_data (dict): A dictionary representing the node profile,
                                  typically loaded from JSON or received over the network.
                                  Expected to contain 'node_id', 'cv' (list of dicts),
                                  'node_specification' (dict), 'peer_id', 'peer_addresses'
                                  and other profile keys.

        Returns:
            NodeProfile: A new instance of NodeProfile populated from the dictionary.

        Raises:
            ValueError: If 'node_id' is missing in the input dictionary.
            TypeError: If the 'cv' data is present but not a list.
        """

        # Ensure essential 'node_id' is present
        node_id = combined_data.get('static').get('node_id')
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
    def _get_os_spec():
        """Extracts operating system information."""
        return platform.platform()

    # Get cpu information
    @staticmethod
    def _get_cpu_info():
        """Extracts CPU core information."""
        try:
            return {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True)
            }
        except Exception as e:
            print(f"Error getting CPU info: {e}")
            return {'physical_cores': None, 'logical_cores': None}

    # Get memory information
    @staticmethod
    def _get_memory_info():
        """Extracts memory information in GB."""
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
            print(f"Error getting memory info: {e}")
            return {'total': 0.0, 'available': 0.0, 'used': 0.0}

    # Get public ip address
    @staticmethod
    def _get_public_ip_address() -> str | None:
        """Attempts to retrieve the public IP address using an external web service.
        Uses multiple services as fallbacks.
        Returns the public IP address string or None if retrieval fails.
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

                # Make a GET request to the service URL with a timeout
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

                # Catch any request-related errors (e.g., network issues, timeout, bad status)
                continue  # Try the next service on error

            except Exception:

                # Catch any other unexpected errors
                continue  # Try the next service on error

        return 'Public IP not available.'  # Return None if all services fail

    # Get guessed location based on IP address
    def _get_geolocation_from_ip(self, ip_address):
        """Retrieves geolocation data (same as before)."""

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
        """Gathers current system specifications.
        """
        cpu_info = self._get_cpu_info()
        memory_info = self._get_memory_info()

        return {
            'timestamp': datetime.datetime.now(timezone.utc).isoformat(),
            'os': self._get_os_spec(),
            'cpu_cores': cpu_info.get('physical_cores'),
            'logical_cpus': cpu_info.get('logical_cores'),
            'memory_gb': memory_info.get('total'),
            'memory_avail': memory_info.get('available'),
            'memory_used': memory_info.get('used'),
            'public_ip_address': self._get_public_ip_address(),
            'guessed_location': self._get_geolocation_from_ip(self._get_public_ip_address()),
        }

    def _fill_missing_specs(self):
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
        """Checks current specs against saved specs. Updates profile data."""

        current_specs = self._get_current_specs()
        specs_changed = False

        if update_only:
            self._profile_data['dynamic'] |= current_specs
        else:
            saved_specs = self._profile_data['dynamic'].copy()
            change_details = []

            if saved_specs is None:

                # No previous specification exists, capture the current one
                self._profile_data['dynamic'] |= current_specs
                specs_changed = True
                change_details.append("Initial specification captured")

            else:

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
                    print(f"Specs changed for '{self._profile_data['static']['node_id']}': {change_summary}")

        self._profile_last_updated = datetime.datetime.now(timezone.utc)  # Mark profile as checked/updated

        return specs_changed

    # Get profile data as dict: cv, dynamic_profile, static_profile
    def get_static_profile(self) -> dict:
        return self._profile_data['static']

    def get_dynamic_profile(self) -> dict:
        return self._profile_data['dynamic']

    def get_cv(self):
        return self._profile_data['cv']

    def get_all_profile(self):
        return self._profile_data

    def mark_change_in_connections(self):
        self._connections_updated = True

    def unmark_change_in_connections(self):
        self._connections_updated = False

    def connections_changed(self):
        return self._connections_updated

    def verify_cv_hash(self, cv_hash: str):
        computed_hash = hashlib.blake2b(json.dumps(self._profile_data['cv']).encode("utf-8"),
                                        digest_size=16).hexdigest()
        return cv_hash == computed_hash, (cv_hash, computed_hash)
