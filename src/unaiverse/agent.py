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
import copy
import json
import uuid
import torch
from unaiverse.dataprops import DataProps
from unaiverse.agent_basics import AgentBasics
from unaiverse.streams import BufferedDataStream
from unaiverse.networking.p2p.messages import Msg


class Agent(AgentBasics):
    """This class contains those basic actions that can be performed by every agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Status variables (assumed to start with "_"): Agent exchanges
        self._available = True  # It will be automatically set/changed during the agent's life
        self._found_agents = set()  # Peer IDs discovered
        self._valid_cmp_agents = set()  # Agents for which the last evaluation was positive
        self._engaged_agents = set()
        self._agents_who_completed_what_they_were_asked = set()
        self._agents_who_were_asked = set()
        self._eval_results = {}

        # Status variables (assumed to start with "_"): Recordings
        self._last_recorded_stream_num = 1
        self._last_recorded_stream_dict = None
        self._last_recording_stream_dict = None

        # Status variables (assumed to start with "_"): Playlist
        self._preferred_streams = []  # List of preferred streams
        self._cur_preferred_stream = 0  # ID of the current preferred stream from the list
        self._repeat = 1  # Number of repetitions of the playlist

    def remove_peer_from_agent_status_attrs(self, peer_id):
        super().remove_peer_from_agent_status_attrs(peer_id)
        self._available = len(self._engaged_agents) == 0

    def reset_agent_status_attrs(self):
        super().reset_agent_status_attrs()  # this sets status vars to [], {}, 0, 0., False, in function of their type
        self._available = True
        self._repeat = 1
        self._last_recorded_stream_num = 1

    def set_next_action(self, agent: str | None, action: str, args: dict | None = None, ref_uuid: str | None = None):
        """Try to tell another agent what is the next action it should run.

        Args:
            agent: The ID of the agent to send the action to or a valid wildcard like "<valid_cmp>" for a set of agents
                (if None the agents in self._engaged_agents will be considered).
            action: The name of the action to be executed by the agent.
            args: A dictionary of arguments for the action. Defaults to None.
            ref_uuid: An optional UUID for referencing the action. Defaults to None.

        Returns:
            True if the action was successfully sent to the target agent or to at least one of the
            involved agents (wildcard case).
        """

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        at_least_one_completed = False
        _, private_peer_id = self.get_peer_ids()
        for _peer_id in involved_agents:
            ret = self._node_conn.send(_peer_id, channel_trail=None,
                                       content={"action_name": action, "args": args, "uuid": ref_uuid},
                                       content_type=Msg.ACTION_REQUEST)
            at_least_one_completed = at_least_one_completed or ret
            self.deb(f"[set_next_action] {self._node_name} sent action: {action}, with args: {args}, "
                     f"and result of sending is {ret}")
        return at_least_one_completed

    def send_engagement(self):
        """Offer engagement to the agents whose identifiers are in self._found_agents.

        Returns:
            True if engagement requests were successfully sent to at least one found agent, False otherwise.
        """
        at_least_one_sent = False

        if len(self._found_agents) > 0:
            self.out(f"Sending engagement request to {', '.join([x for x in self._found_agents])}")
        my_role_str = self._node_profile.get_dynamic_profile()['connections']['role']
        for found_agent in self._found_agents:
            if self.set_next_action(found_agent, action="get_engagement",
                                    args={"sender_role": my_role_str}):
                at_least_one_sent = True
            else:
                self.err(f"Unable to send engagement to {found_agent}")

        return at_least_one_sent

    def get_engagement(self, acceptable_role: str | None = None, sender_role: str | None = None,
                       _requester: str | None = None):
        """Receive engagement from another agent whose authority is in the specified range.

        Args:
            acceptable_role: The role that the sender must have for engagement to be accepted. Defaults to None.
            sender_role: The role of the agent sending the engagement request. Defaults to None.
            _requester: The ID of the agent requesting engagement (automatically set by the action calling routine)

        Returns:
            True if the engagement was successfully received and confirmed, False otherwise.
        """
        self.out(f"Getting engagement from {_requester}, whose role is {sender_role} (looking for {acceptable_role})")
        if _requester not in self.world_agents and _requester not in self.world_masters:
            self.err(f"Unknown agent: {_requester}")
            return False

        if sender_role is None:
            self.err(f"Unknown role of {_requester}")
            return False

        # Confirming
        if self._available:
            acceptable_role_int = self.ROLE_STR_TO_BITS[acceptable_role]
            if "~" not in acceptable_role:
                sender_role_int = (self.ROLE_STR_TO_BITS[sender_role] >> 2) << 2
            else:
                sender_role_int = self.ROLE_STR_TO_BITS[sender_role]

            if acceptable_role_int == sender_role_int:
                if self.set_next_action(_requester, "got_engagement"):
                    self._engaged_agents.add(_requester)

                    # Marking this agent as not available since it engaged with another one
                    self._available = False
                    return True
                else:
                    self.err(f"Unable to confirm engagement to {_requester}")
                    return False
            else:
                self.err(f"Cannot engage to {_requester}")
                return False
        else:
            self.err(f"Cannot engage to {_requester}")
            return False

    def got_engagement(self, _requester: str | None = None):
        """Confirm an engagement.

        Args:
            _requester: The ID of the agent confirming the engagement (automatically set by the action calling routine).

        Returns:
            True if the engagement was successfully confirmed, False otherwise.
        """
        self.out(f"Confirming engagement with {_requester}")
        if _requester in self._found_agents:
            self._engaged_agents.add(_requester)

            # Marking this agent as not available since it engaged with another one
            self._available = False

            # Removing the engaged agent from the list of found agents, to avoid sending him another engagement request
            self._found_agents.discard(_requester)
            return True
        else:
            self.err(f"Unable to confirm engagement with {_requester}")
            return False

    def send_disengagement(self, send_disconnection_too: bool = False):
        """Ask for disengagement.

        Args:
            send_disconnection_too: Whether to send a disconnect-suggestion together with the disengagement.

        Returns:
            True if disengagement requests were successfully sent to at least one engaged agent, False otherwise.
        """
        at_least_one_sent = False

        if len(self._engaged_agents) > 0:
            self.out(f"Sending disengagement request to {', '.join([x for x in self._engaged_agents])}")
        for agent in self._engaged_agents:
            if self.set_next_action(agent, action="get_disengagement", args={"disconnect_too": send_disconnection_too}):
                at_least_one_sent = True
            else:
                self.err(f"Unable to send disengagement to {agent}")

        return at_least_one_sent

    def get_disengagement(self, disconnect_too: bool = False, _requester: str | None = None):
        """Get a disengagement request from an agent.

        Args:
            disconnect_too: Whether to disconnect the agent who sent the disengagement.
            _requester: The ID of the agent requesting disengagement. Defaults to None.

        Returns:
            True if the disengagement request was successfully processed, False otherwise.
        """
        self.out(f"Getting a disengagement request from {_requester}")
        if _requester not in self.world_agents and _requester not in self.world_masters:
            self.err(f"Unknown agent: {_requester}")
            return False

        if _requester not in self._engaged_agents:
            self.err(f"Not previously engaged to {_requester}")
            return False

        if disconnect_too:
            self._node_purge_fcn(_requester)

        self._engaged_agents.discard(_requester)  # Remove if present

        # Marking this agent as available if not engaged to any agent
        self._available = len(self._engaged_agents) == 0
        return True

    def disengage_all(self):
        """Disengage all the previously engaged agents.

        Returns:
            True if the disengagement procedure was successfully executed, False otherwise.
        """
        self.out(f"Disengaging all agents")
        self._engaged_agents = set()

        # Marking this agent as available
        self._available = True
        return True

    def disconnect_by_role(self, role: str | list[str]):
        """Disconnects from all agents that match a specified role.
        It finds the agents and calls the node's purge function on each.

        Args:
            role: A string or list of strings representing the role(s) of agents to disconnect from.

        Returns:
            Always True.
        """
        self.out(f"Disconnecting agents with role: {role}")
        if self.find_agents(role):
            found_agents = copy.deepcopy(self._found_agents)
            for agent in found_agents:
                self._node_purge_fcn(agent)  # This will also call remove_agent, that will call remove_streams
        return True

    def disconnected(self, agent: str | None = None, delay: float = -1.):
        """Checks if a specific set of agents (by ID or wildcard) are no longer connected to the agent.
        It returns False if any of the specified agents are still connected.

        Args:
            agent: The ID of the agent or a wildcard to check.
            delay: The time (seconds) to be spent in the current state before actually considering this action.

        Returns:
            True if all involved agents are disconnected, False otherwise.

        """
        assert delay is not None, "Missing basic action information"

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.out(f"Checking if all these agents are not connected to me anymore: {involved_agents}")
        all_disconnected = True
        for agent in involved_agents:
            if agent in self.world_agents or agent in self.public_agents or agent in self._node_agents_waiting\
                    or self._node_conn.is_connected(agent):
                all_disconnected = False
                break
        return all_disconnected

    def received_some_asked_data(self, processing_fcn: str | None = None):
        """Checks if any of the agents that were previously asked for data (e.g., via `ask_gen`) have sent a stream
        sample back. Optionally, it can process the received data with a specified function.

        Args:
            processing_fcn: The name of a function to process the received data.

        Returns:
            True if at least one data sample was received, False otherwise.
        """
        _processing_fcn = None
        if processing_fcn is not None:
            if hasattr(self, processing_fcn):
                _processing_fcn = getattr(self, processing_fcn)
                if not callable(_processing_fcn):
                    _processing_fcn = None
            if _processing_fcn is None:
                self.err(f"Processing function not found: {processing_fcn}")

        got_something = False
        for agent in self._agents_who_were_asked:
            net_hash_to_stream_dict = self.find_streams(agent, "processor")
            for stream_dict in net_hash_to_stream_dict.values():
                for stream_obj in stream_dict.values():
                    if not stream_obj.props.is_public():
                        data = stream_obj.get("received_some_asked_data")
                        data_tag = stream_obj.get_tag()

                        if data is not None:
                            if _processing_fcn is None:
                                return True
                            else:
                                got_something = True
                                _processing_fcn(agent, stream_obj.props, data, data_tag)
        return got_something

    def nop(self, message: str | None = None, delay: float = -1.):
        """Do nothing.

        Args:
            message: An optional message to print. Defaults to None.
            delay: The time (seconds) to be spent in the current state before actually considering this action.

        Returns:
            Always True.
        """
        assert delay is not None, "Missing basic action information"
        if message is not None:
            self.out(message)
        return True

    def wait_for_actions(self, agent: str, from_state: str, to_state: str, wait: bool):
        """Lock or unlock every action between a pair of states in the state machine of a target agent.

        Args:
            agent: The ID of the agent to send the action locking request to, or a valid wildcard like "<valid_cmp>"
                for a set of agents (if None the agents in self._engaged_agents will be considered).
            from_state: The starting state of the actions to be locked/unlocked.
            to_state: The ending state of the actions to be locked/unlocked.
            wait: A boolean indicating whether to wait for the actions to complete (wait == !ready).

        Returns:
            True if the request was successfully sent to at least one involved agent, False otherwise.
        """

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        at_least_one_completed = False
        for _agent in involved_agents:
            self.out(f"Telling {_agent} to alter his HSM {from_state} -> {to_state} (wait: {wait}) "
                     f"by calling method 'wait_for_actions' on it")
            ret = self._node_conn.send(_agent, channel_trail=None,
                                       content={'method': 'wait_for_actions', 'args': (from_state, to_state, wait)},
                                       content_type=Msg.HSM)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def ask_gen(self, agent: str | None = None, u_hashes: list[str] | None = None,
                samples: int = 100, time: float = -1., timeout: float = -1., ask_uuid: str | None = None,
                ignore_uuid: bool = False):
        """Asking for generation.

        Args:
            agent: The ID of the agent to ask for generation, or a valid wildcard like "<valid_cmp>"
                for a set of agents (if None the agents in self._engaged_agents will be considered).
            u_hashes: A list of input stream hashes for generation. Defaults to None.
            samples: The number of samples to generate. Defaults to 100.
            time: The time duration for generation. Defaults to -1.
            timeout: The timeout for the generation request. Defaults to -1.
            ask_uuid: Specify the UUID of the action (if None - default -, it is randomly generated).
            ignore_uuid: Force a None UUID instead of generating a random one.

        Returns:
            True if the generation request was successfully sent to at least one involved agent, False otherwise.
        """
        assert samples is not None and time is not None and timeout is not None, "Missing basic action information"

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        self.deb(f"[ask_gen] Involved_agents: {involved_agents}")

        if len(involved_agents) == 0:
            self.deb(f"[ask_gen] No involved agents, action ask_gen returns False")
            return False

        # Create a copy of the input hashes, normalizing them in the appropriate way
        u_hashes_copy: list[str | None] = [None] * len(u_hashes)
        for i in range(len(u_hashes_copy)):
            if u_hashes_copy[i] == "<playlist>":

                # From <playlist> to the current element of the playlist
                u_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:

                # From a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                u_hashes_copy[i] = self.user_stream_hash_to_net_hash(u_hashes[i])

        # Generate a new UUID for this request
        ref_uuid = uuid.uuid4().hex[0:8] if ask_uuid is None else ask_uuid
        if ignore_uuid:
            ref_uuid = None

        # If the input streams are all owned by this agent, discard UUID
        all_owned = True
        for i in range(len(u_hashes_copy)):
            if u_hashes_copy[i] not in self.owned_streams:
                all_owned = False
                break
        if not all_owned:
            ref_uuid = None

        for i in range(len(u_hashes_copy)):

            # If there are our own streams involved, and they are buffered, let's plan to restart them when we will
            # start sending them through the net: moreover, let's set the local stream UUID appropriately to
            # the generated UUID
            if u_hashes_copy[i] in self.owned_streams:
                stream_dict = self.known_streams[u_hashes_copy[i]]
                for stream_name, stream_obj in stream_dict.items():

                    # Plan to restart buffered streams
                    if isinstance(stream_obj, BufferedDataStream):
                        stream_obj.plan_restart_before_next_get(requested_by="send_stream_samples")

                    # Activate the stream (if it was off)
                    stream_obj.enable()

                    # Set UUID to the generated one
                    stream_obj.set_uuid(ref_uuid=ref_uuid, expected=False)
                    stream_obj.set_uuid(ref_uuid=None, expected=True)

        self.deb(f"[ask_gen] Input streams u_hashes: {u_hashes_copy}")

        self.out(f"Asking {', '.join(involved_agents)} to generate signal given {u_hashes_copy} (ref_uuid: {ref_uuid})")
        self._agents_who_completed_what_they_were_asked = set()
        self._agents_who_were_asked = set()
        correctly_asked = []
        for peer_id in involved_agents:
            ret = self.__ask_gen_or_learn(for_what="gen", agent=peer_id,
                                          u_hashes=u_hashes_copy,
                                          yhat_hashes=None,
                                          samples=samples, time=time, timeout=timeout, ref_uuid=ref_uuid)
            self.deb(f"[ask_gen] Asking {peer_id} returned {ret}")
            if ret:
                correctly_asked.append(peer_id)

        # Preparing the buffered stream where to store data, if needed
        if len(correctly_asked) > 0:

            # Saving
            self.last_ref_uuid = ref_uuid

            # For each agent that we involve in this request....
            for peer_id in correctly_asked:

                # Finding the streams generated by the processor of the agent we asked to generate
                processor_streams = self.find_streams(peer_id, name_or_group="processor")

                # For each stream generated by the processor of the agent we asked to generate...
                for net_hash, stream_dict in processor_streams.items():

                    # Set the appropriate UUID to the one we created in this method
                    for stream in stream_dict.values():
                        stream.set_uuid(None, expected=False)
                        stream.set_uuid(ref_uuid, expected=True)  # Setting the "expected" one

        self.deb(f"[ask_gen] Overall, the action ask_gen will return {len(correctly_asked) > 0}")
        return len(correctly_asked) > 0

    def do_gen(self, u_hashes: list[str] | None = None,
               samples: int = 100, time: float = -1., timeout: float = -1.,
               _requester: str | list | None = None, _request_time: float = -1., _request_uuid: str | None = None,
               _completed: bool = False) -> bool:
        """Generate a signal.

        Args:
            u_hashes: A list of input stream hashes for generation. Defaults to None.
            samples: The number of samples to generate. Defaults to 100.
            time: The max time duration for whole generation process. Defaults to -1.
            timeout: The timeout for generation attempts: if calling the generate action fails for more than "timeout"
            seconds, it is declared as complete. Defaults to -1.
            _requester: The ID of the agent who requested generation (automatically set by the action calling routine).
            _request_time: The time the generation was requested (automatically set by the action calling routine).
            _request_uuid: The UUID of the generation request (automatically set by the action calling routine).
            _completed: A boolean indicating if the generation is already completed (automatically set by the action
                calling routine). This will tell that it is time to run a final procedure.

        Returns:
            True if the signal generation was successful, False otherwise.
        """
        assert samples is not None and time is not None and timeout is not None, "Missing basic action information"

        self.deb(f"[do_gen] Samples: {samples}, time: {time}, timeout: {timeout}, "
                 f"requester: {_requester}, request_time: {_request_time}, request_uuid: {_request_uuid}, "
                 f"completed: {_completed}")

        if _requester is not None:
            if isinstance(_requester, list):
                for _r in _requester:
                    if self.behaving_in_world():
                        if _r not in self.world_agents and _requester not in self.world_masters:
                            self.err(f"Unknown agent: {_r} in list {_requester} (fully skipping generation)")
                            return False
                    else:
                        if _r not in self.public_agents:
                            self.err(f"Unknown agent: {_r} in list {_requester} (fully skipping generation)")
                            return False
            else:
                if self.behaving_in_world():
                    if _requester not in self.world_agents and _requester not in self.world_masters:
                        self.err(f"Unknown agent: {_requester} (fully skipping generation)")
                        return False
                else:
                    if _requester not in self.public_agents:
                        self.err(f"Unknown agent: {_requester} (fully skipping generation)")
                        return False

        # Check what is the step ID of the multistep action
        k = self.get_action_step()

        # In the first step of this action, we change the UUID of the local stream associated to the input data we will
        # use to handle this action, setting expectations to avoid handling tags of old data
        if k == 0:

            # Warning: we are not normalizing the hashes, we should do it if this action is called directly
            if u_hashes is not None:
                for net_hash in u_hashes:
                    if net_hash in self.known_streams:
                        for stream_name, stream_obj in self.known_streams[net_hash].items():

                            # If the data arrived before this action, then the UUID is already set, and here there is
                            # no need to do anything; if the data has not yet arrived (common case) ...
                            if stream_obj.get_uuid(expected=False) != _request_uuid:
                                stream_obj.set_uuid(None, expected=False)  # Clearing UUID
                                stream_obj.set_uuid(_request_uuid, expected=True)  # Setting expectations
                    else:
                        self.out(f"Unknown stream mentioned in u_hashes: {net_hash}")
                        return False

        if not _completed:
            self.out(f"Generating signal")
            ret = self.__process_streams(u_hashes=u_hashes, yhat_hashes=None, learn=False,
                                         recipient=_requester, ref_uuid=_request_uuid)
            if not ret:
                self.out(f"Generating signal failed")
            else:
                if not self.is_multi_steps_action():
                    self.out(f"Completing signal generation (degenerate single-step case of a multi-step action")
                    ret = self.__complete_do(do_what="gen", peer_id_who_asked=_requester, all_hashes=u_hashes,
                                             send_back_confirmation=False)
                    if not ret:
                        self.out(f"Completing signal generation failed")
            return ret
        else:
            self.out(f"Completing signal generation")
            ret = self.__complete_do(do_what="gen", peer_id_who_asked=_requester, all_hashes=u_hashes)
            if not ret:
                self.out(f"Completing signal generation failed")
            return ret

    def done_gen(self, _requester: str | None = None):
        """This is a way to get back the confirmation of a completed generation.

        Args:
            _requester: The ID of the agent who completed the generation. Defaults to None.

        Returns:
            True if the generation confirmation was successfully handled by this agent, False is something went wrong.
        """
        self.out(f"Agent {_requester} finished generation")

        # Searching for the processor-streams of the agent who generated data
        processor_streams = self.find_streams(_requester, name_or_group="processor")
        if processor_streams is None or len(processor_streams) == 0:
            self.err("Unexpected confirmation of finished generation")
            return False

        # Remembering that the agent that invoked this action is the one who generated the data, and what he generated
        # could be used in future action (for example, in evaluation processes)
        self._agents_who_completed_what_they_were_asked.add(_requester)

        # Clearing the UUID of the local streams associated to the agent who generated
        for net_hash, stream_dict in processor_streams.items():
            for stream_obj in stream_dict.values():
                stream_obj.set_uuid(None, expected=False)
                stream_obj.set_uuid(None, expected=True)

        # If one or more of my streams where used as arguments of the generation request I did (ask_gen), then their
        # UUID must be cleared...we clear them all
        for net_hash, stream_dict in self.owned_streams.items():
            for stream_obj in stream_dict.values():
                if stream_obj.props.is_public() != self.behaving_in_world():
                    stream_obj.set_uuid(None, expected=False)
                    stream_obj.set_uuid(None, expected=True)
        return True

    def ask_learn(self, agent: str | None = None,
                  u_hashes: list[str] | None = None, yhat_hashes: list[str] | None = None,
                  samples: int = 100, time: float = -1., timeout: float = -1., ask_uuid: str | None = None,
                  ignore_uuid: str | None = None):
        """Asking for learning to generate.

        Args:
            agent: The ID of the agent to ask for generation, or a valid wildcard like "<valid_cmp>"
                for a set of agents (if None the agents in self._engaged_agents will be considered).
            u_hashes: A list of input stream hashes for inference. Defaults to None.
            yhat_hashes: A list of target stream hashes to be used for loss computation. Defaults to None.
            samples: The number of samples to learn from. Defaults to 100.
            time: The time duration for generation. Defaults to -1.
            timeout: The timeout for the generation request. Defaults to -1.
            ask_uuid: Specify the action UUID (default = None, i.e., it is automatically generated).
            ignore_uuid: If Trie, the UUID is fully ignored (i.e, forced to None).

        Returns:
            True if the learning request was successfully sent to at least one involved agent, False otherwise.
        """
        assert samples is not None and time is not None and timeout is not None, "Missing basic action information"

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        self.deb(f"[ask_learn] Involved agents: {involved_agents}")

        if len(involved_agents) == 0:
            self.deb(f"[ask_learn] No involved agents, action will return False")
            return False

        # Create a copy of the input hashes, normalizing them in the appropriate way
        u_hashes_copy = [x for x in u_hashes]
        for i in range(len(u_hashes_copy)):
            if u_hashes_copy[i] == "<playlist>":

                # From <playlist> to the current element of the playlist
                u_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:

                # From a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                u_hashes_copy[i] = self.user_stream_hash_to_net_hash(u_hashes_copy[i])

        # Create a copy of the target hashes, normalizing them in the appropriate way
        yhat_hashes_copy = [x for x in yhat_hashes]
        for i in range(len(yhat_hashes_copy)):
            if yhat_hashes_copy[i] == "<playlist>":

                # From <playlist> to the current element of the playlist
                yhat_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:

                # From a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                yhat_hashes_copy[i] = self.user_stream_hash_to_net_hash(yhat_hashes_copy[i])

        # Generate a new UUID for this request
        ref_uuid = uuid.uuid4().hex[0:8] if ask_uuid is None else ask_uuid
        if ignore_uuid:
            ref_uuid = None

        # If the input streams are all owned by this agent, discard UUID
        all_owned = True
        for i in range(len(u_hashes_copy)):
            if u_hashes_copy[i] not in self.owned_streams:
                all_owned = False
                break
        if all_owned:
            for i in range(len(yhat_hashes_copy)):
                if yhat_hashes_copy[i] not in self.owned_streams:
                    all_owned = False
                    break
        if not all_owned:
            ref_uuid = None

        for i in range(len(u_hashes_copy)):

            # If there are our own streams involved, and they are buffered, let's plan to restart them when we will
            # start sending them through the net: moreover, let's set the local stream UUID appropriately to
            # the generated UUID
            if u_hashes_copy[i] in self.owned_streams:
                stream_dict = self.known_streams[u_hashes_copy[i]]
                for stream_name, stream_obj in stream_dict.items():

                    # Plan to restart buffered streams
                    if isinstance(stream_obj, BufferedDataStream):
                        stream_obj.plan_restart_before_next_get(requested_by="send_stream_samples")

                    # Activate the stream (if it was off)
                    stream_obj.enable()

                    # Set UUID to the generated one
                    stream_obj.set_uuid(ref_uuid=ref_uuid, expected=False)
                    stream_obj.set_uuid(ref_uuid=None, expected=True)

        for i in range(len(yhat_hashes_copy)):

            # If there are our own streams involved, and they are buffered, let's plan to restart them when we will
            # start sending them through the net: moreover, let's set the local stream UUID appropriately to
            # the generated UUID
            if yhat_hashes_copy[i] in self.owned_streams:
                stream_dict = self.known_streams[yhat_hashes_copy[i]]
                for stream_name, stream_obj in stream_dict.items():

                    # Plan to restart buffered streams
                    if isinstance(stream_obj, BufferedDataStream):
                        stream_obj.plan_restart_before_next_get(requested_by="send_stream_samples")

                    # Activate the stream (if it was off)
                    stream_obj.enable()

                    # Set UUID to the generated one
                    stream_obj.set_uuid(ref_uuid=ref_uuid, expected=False)
                    stream_obj.set_uuid(ref_uuid=None, expected=True)

        self.out(f"Asking {', '.join(involved_agents)} to learn to generate signal {yhat_hashes_copy}, "
                 f"given {u_hashes_copy} (ref_uuid: {ref_uuid})")
        self._agents_who_completed_what_they_were_asked = set()
        self._agents_who_were_asked = set()
        correctly_asked = []
        for peer_id in involved_agents:
            ret = self.__ask_gen_or_learn(for_what="learn", agent=peer_id,
                                          u_hashes=u_hashes_copy,
                                          yhat_hashes=yhat_hashes_copy,
                                          samples=samples, time=time, timeout=timeout, ref_uuid=ref_uuid)
            self.deb(f"[ask_learn] Asking {peer_id} returned {ret}")
            if ret:
                correctly_asked.append(peer_id)

        # Preparing the buffered stream where to store data, if needed
        if len(correctly_asked) > 0:

            # Saving
            self.last_ref_uuid = ref_uuid

            # For each agent that we involve in this request....
            for peer_id in correctly_asked:

                # Finding the streams generated by the processor of the agent we asked to generate
                processor_streams = self.find_streams(peer_id, name_or_group="processor")

                # For each stream generated by the processor of the agent we asked to generate...
                for net_hash, stream_dict in processor_streams.items():

                    # Set the appropriate UUID to the one we created in this method
                    for stream in stream_dict.values():
                        stream.set_uuid(None, expected=False)
                        stream.set_uuid(ref_uuid, expected=True)  # Setting the "expected" one

        self.deb(f"[ask_learn] Overall the action ask_learn will return {len(correctly_asked) > 0}")
        return len(correctly_asked) > 0

    def do_learn(self, yhat_hashes: list[str] | None = None, u_hashes: list[str] | None = None,
                 samples: int = 100, time: float = -1., timeout: float = -1.,
                 _requester: str | None = None, _request_time: float = -1., _request_uuid: str | None = None,
                 _completed: bool = False) -> bool:
        """Learn to generate a signal.

        Args:
            yhat_hashes: A list of target stream hashes to be used for loss computation. Defaults to None.
            u_hashes: A list of input stream hashes for inference. Defaults to None.
            samples: The number of samples to learn from. Defaults to 100.
            time: The max time duration of the learning procedure. Defaults to -1.
            timeout: The timeout for learning attempts: if calling the learning action fails for more than "timeout"
            seconds, it is declared as complete. Defaults to -1.
            _requester: The ID of the agent who requested learning (automatically set by the action calling routine).
            _request_time: The time learning was requested (automatically set by the action calling routine).
            _request_uuid: The UUID of the learning request (automatically set by the action calling routine).
            _completed: A boolean indicating if the learning is already completed (automatically set by the action
                calling routine). This will tell that it is time to run a final procedure.

        Returns:
            True if the signal generation was successful, False otherwise.
        """
        assert samples is not None and time is not None and timeout is not None, "Missing basic action information"

        self.deb(f"[do_learn] samples: {samples}, time: {time}, timeout: {timeout}, "
                 f"requester: {_requester}, request_time: {_request_time}, request_uuid: {_request_uuid} "
                 f"completed: {_completed}")

        if _requester not in self.world_agents and _requester not in self.world_masters:
            self.err(f"Unknown agent: {_requester}")
            return False

        # Check what is the step ID of the multistep action
        k = self.get_action_step()

        # In the first step of this action, we change the UUID of the local stream associated to the input data we will
        # use to handle this action, setting expectations to avoid handling tags of old data
        if k == 0:

            # Warning: we are not normalizing the hashes, we should do it if this action is called directly
            if u_hashes is not None:
                for net_hash in u_hashes:
                    if net_hash in self.known_streams:
                        for stream_obj in self.known_streams[net_hash].values():

                            # If the data arrived before this action, then the UUID is already set, and here there is
                            # no need to do anything; if the data has not yet arrived (common case) ...
                            if stream_obj.get_uuid(expected=False) != _request_uuid:
                                stream_obj.set_uuid(None, expected=False)  # Clearing UUID
                                stream_obj.set_uuid(_request_uuid, expected=True)  # Setting expectations

            # Warning: we are not normalizing the hashes, we should do it if this action is called directly
            if yhat_hashes is not None:
                for net_hash in yhat_hashes:
                    if net_hash in self.known_streams:
                        for stream_obj in self.known_streams[net_hash].values():
                            if stream_obj.get_uuid(expected=False) != _request_uuid:
                                stream_obj.set_uuid(None, expected=False)  # Clearing UUID
                                stream_obj.set_uuid(_request_uuid, expected=True)  # Setting expectations

        if not _completed:
            self.out(f"Learning to generate signal {yhat_hashes}")
            ret = self.__process_streams(u_hashes=u_hashes, yhat_hashes=yhat_hashes, learn=True,
                                         recipient=_requester, ref_uuid=_request_uuid)
            if not ret:
                self.out(f"Learning to generate signal {yhat_hashes} failed")
            return ret
        else:
            self.out(f"Completing learning to generate signal {yhat_hashes}")
            all_hashes = (u_hashes if u_hashes is not None else []) + (yhat_hashes if yhat_hashes is not None else [])
            ret = self.__complete_do(do_what="learn", peer_id_who_asked=_requester, all_hashes=all_hashes)
            if not ret:
                self.out(f"Completing learning to generate signal {yhat_hashes} failed")
            return ret

    def done_learn(self, _requester: str | None = None):
        """This is a way to get back the confirmation of a completed learning procedure.

        Args:
            _requester: The ID of the agent who completed the learning procedure. Defaults to None.

        Returns:
            True if the learning-complete confirmation was successfully handled by this agent, False otherwise.
        """
        self.out(f"Agent {_requester} finished learning")
        self._agents_who_completed_what_they_were_asked.add(_requester)

        # Searching for the processor-streams of the agent who generated the (inference) data
        processor_streams = self.find_streams(_requester, name_or_group="processor")
        if processor_streams is None or len(processor_streams) == 0:
            self.err("Unexpected confirmation of finished learning")
            return False

        # Warning: differently from the case of done_gen, we are not considering the streams generated by the
        # learning agents as something we could use for evaluation (this might be changed in the future)

        # Clearing the UUID of the local streams associated to the agent who learned
        for net_hash, stream_dict in processor_streams.items():
            for stream_obj in stream_dict.values():
                stream_obj.set_uuid(None, expected=False)
                stream_obj.set_uuid(None, expected=True)

        # If one or more of my streams where used as arguments of the learning request I did (ask_learn), then their
        # UUID must be cleared...we clear them all
        for net_hash, stream_dict in self.owned_streams.items():
            for stream_obj in stream_dict.values():
                if stream_obj.props.is_public() != self.behaving_in_world():
                    stream_obj.set_uuid(None, expected=False)
                    stream_obj.set_uuid(None, expected=True)
        return True

    def all_asked_finished(self):
        """Checks if all agents that were previously asked to perform a task (e.g., generate or learn) have sent a
        completion confirmation. It compares the set of agents asked with the set of agents that have completed
        the task.

        Returns:
            True if all agents are done, False otherwise.
        """
        return self._agents_who_were_asked == self._agents_who_completed_what_they_were_asked

    def all_engagements_completed(self):
        """Checks if all engagement requests that were sent have been confirmed. It returns True if there are no agents
        remaining in the `_found_agents` list, implying all have been engaged with or discarded.

        Returns:
            True if all engagements are complete, False otherwise.

        """
        return len(self._found_agents) == 0

    def agents_are_waiting(self):
        """Checks if there are any agents who have connected but have not yet been fully processed or added to the
        agent's known lists. This indicates that new agents are waiting to be managed.

        Returns:
            True if there are waiting agents, False otherwise.
        """
        self.out(f"Current set of {len(self._node_agents_waiting)} connected peer IDs non managed yet: "
                 f"{self._node_agents_waiting}")
        for found_agent in self._found_agents:
            if found_agent in self._node_agents_waiting:
                return True
        return False

    def ask_subscribe(self, agent: str | None = None,
                      stream_hashes: list[str] | None = None, unsubscribe: bool = False):
        """Requests a remote agent or a group of agents to subscribe to or unsubscribe from a list of specified PubSub
        streams. It normalizes the stream hashes and sends an action request containing the stream properties.

        Args:
            agent: The target agent's ID or a wildcard.
            stream_hashes: A list of streams to subscribe to or unsubscribe from.
            unsubscribe: A boolean to indicate if it's an unsubscription request.

        Returns:
            True if the request was sent to at least one agent, False otherwise.
        """

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        self.deb(f"[ask_subscribe] Involved_agents: {involved_agents}")

        if len(involved_agents) == 0:
            self.deb(f"[ask_subscribe] No involved agents, action ask_gen returns False")
            return False

        # Create a copy of the stream hashes, normalizing them in the appropriate way
        stream_hashes_copy: list[str | None] = [None] * len(stream_hashes)
        for i in range(len(stream_hashes_copy)):
            if stream_hashes_copy[i] == "<playlist>":

                # From <playlist> to the current element of the playlist
                stream_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:

                # From a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                stream_hashes_copy[i] = self.user_stream_hash_to_net_hash(stream_hashes[i])

        # Getting properties
        stream_owners = []
        stream_props = []
        for i in range(len(stream_hashes_copy)):
            stream_dict = self.known_streams[stream_hashes_copy[i]]
            peer_id = DataProps.peer_id_from_net_hash(stream_hashes_copy[i])
            for name, stream_obj in stream_dict.items():
                stream_owners.append(peer_id)
                stream_props.append(json.dumps(stream_obj.props.to_dict()))

        what = "subscribe to" if not unsubscribe else "unsubscribe from "
        self.out(f"Asking {', '.join(involved_agents)} to {what} {stream_hashes}")
        self._agents_who_completed_what_they_were_asked = set()
        self._agents_who_were_asked = set()
        correctly_asked = []
        for agent in involved_agents:
            if self.set_next_action(agent, action="do_subscribe", args={"stream_owners": stream_owners,
                                                                        "stream_props": stream_props,
                                                                        "unsubscribe": unsubscribe}):
                self._agents_who_were_asked.add(agent)
                ret = True
            else:
                what = "subscribe" if not unsubscribe else "unsubscribe"
                self.err(f"Unable to ask {agent} to {what}")
                ret = False
            self.deb(f"[ask_subscribe] Asking {agent} returned {ret}")
            if ret:
                correctly_asked.append(agent)

        self.deb(f"[ask_subscribe] Overall, the action ask_subscribe (unsubscribe: {unsubscribe})"
                 f" will return {len(correctly_asked) > 0}")
        return len(correctly_asked) > 0

    def do_subscribe(self, stream_owners: list[str] | None = None, stream_props: list[str] | None = None,
                     unsubscribe: bool = False,
                     _requester: str | list | None = None, _request_time: float = -1.):
        """Executes a subscription or unsubscription request received from another agent. It processes the stream
        properties, adds or removes the streams from the agent's known streams, and handles the underlying PubSub topic
        subscriptions.

        Args:
            stream_owners: A list of peer IDs who own the streams.
            stream_props: A list of JSON-serialized stream properties.
            unsubscribe: A boolean to indicate unsubscription.
            _requester: The ID of the requesting agent.
            _request_time: The time the request was made.

        Returns:
            True if the action is successful, False otherwise.
        """
        self.deb(f"[do_subscribe] unsubscribe: {unsubscribe}, "
                 f"stream_owners: {stream_owners}, stream_props: ... ({len(stream_props)} props)")

        if _requester is not None:
            if isinstance(_requester, list):
                for _r in _requester:
                    if self.behaving_in_world():
                        if _r not in self.world_agents and _requester not in self.world_masters:
                            self.err(f"Unknown agent: {_r} in list {_requester} (fully skipping do_subscribe)")
                            return False
                    else:
                        if _r not in self.public_agents:
                            self.err(f"Unknown agent: {_r} in list {_requester} (fully skipping do_subscribe)")
                            return False
            else:
                if self.behaving_in_world():
                    if _requester not in self.world_agents and _requester not in self.world_masters:
                        self.err(f"Unknown agent: {_requester} (fully skipping do_subscribe)")
                        return False
                else:
                    if _requester not in self.public_agents:
                        self.err(f"Unknown agent: {_requester} (fully skipping do_subscribe)")
                        return False
        else:
            self.err("Unknown requester (None)")
            return False

        # Building properties
        props_dicts = []
        props_objs = []
        for i in range(len(stream_props)):
            p_dict = json.loads(stream_props[i])
            props = DataProps.from_dict(p_dict)
            if props.is_pubsub():
                props_dicts.append(p_dict)
                props_objs.append(props)
            else:
                self.err(f"Expecting a pubsub stream, got a stream named {props.get_name()} "
                         f"(group is {props.get_group()}), which is not pubsub")
                return False

        # Adding new streams and subscribing (if compatible with our processor)
        for stream_owner, prop_dict, prop_obj in zip(stream_owners, props_dicts, props_objs):
            if not unsubscribe:
                if not self.add_compatible_streams(peer_id=stream_owner, streams_in_profile=[prop_dict],
                                                   buffered=False, public=False):
                    self.out(f"Unable to add a pubsub stream ({prop_obj.get_name()}) from agent {stream_owner}: "
                             f"no compatible streams were found")
            else:
                if not self.remove_streams(peer_id=stream_owner, name=prop_obj.get_name()):
                    self.out(f"Unable to unsubscribe from pubsub stream ({prop_obj.get_name()}) "
                             f"of agent {stream_owner}")
        return True

    def done_subscribe(self, unsubscribe: bool = False, _requester: str | None = None):
        """Handles the confirmation that a subscription or unsubscription request has been completed by another agent.
        It adds the requester to the set of agents that have completed their asked tasks.

        Args:
            unsubscribe: A boolean indicating if it was an unsubscription.
            _requester: The ID of the agent who completed the task.

        Returns:
            Always True.
        """
        what = "subscribing" if unsubscribe else "unsubscribing"
        self.out(f"Agent {_requester} finished {what}")

        # Remembering that the agent that invoked this action is the one who actually subscribed
        self._agents_who_completed_what_they_were_asked.add(_requester)
        return True

    def record(self, net_hash: str, samples: int = 100, time: float = -1., timeout: float = -1.):
        """Records data from a specified stream into a new, owned `BufferedDataStream`. This is a multistep action
        that captures a sequence of samples over time and then adds the new recorded stream to the agent's profile.

        Args:
            net_hash: The hash of the stream to record.
            samples: The number of samples to record.
            time: The time duration for recording.
            timeout: The timeout for each recording attempt.

        Returns:
            True if a sample was successfully recorded, False otherwise.
        """
        assert samples is not None and time is not None and timeout is not None, "Missing basic action information"

        k = self.get_action_step()

        self.out(f"Recording stream {net_hash}")

        if k == 0:

            # Getting stream(s)
            _net_hash = self.user_stream_hash_to_net_hash(net_hash)  # In case of ambiguity, it yields the first one
            if _net_hash is None:
                self.err(f"Unknown stream {net_hash}")
                return False
            else:
                net_hash = _net_hash

            stream_src_dict = self.known_streams[net_hash]

            # Creating the new recorded stream (same props of the recorded one, just owned now)
            stream_dest_dict = {}
            for name, stream_obj in stream_src_dict.items():
                props = stream_obj.props.clone()
                props.set_group("recorded" + str(self._last_recorded_stream_num))
                stream_dest_dict[name] = BufferedDataStream(props=props, clock=self._node_clock)
            self._last_recorded_stream_dict = stream_dest_dict
            self._last_recording_stream_dict = stream_src_dict

        else:

            # Retrieving the stream(s)
            stream_dest_dict = self._last_recorded_stream_dict
            stream_src_dict = self._last_recording_stream_dict

        # Recording
        for name, stream_obj in stream_src_dict.items():
            x = stream_obj.get(requested_by="record")
            if x is None:
                self.deb("[record] data sample missing, returning False")
                return False
            else:
                self.deb(f"[record] data_tag: {stream_obj.get_tag()}, data_uuid: {stream_obj.get_uuid()}")
            stream_dest_dict[name].set(x, k)  # Saving specific data tags 0, 1, 2, ... #record_steps - 1

        # Updating profile
        if self.is_last_action_step():
            self.deb("[record] last action step detected, finishing")

            # Dummy get to ensure that the next get will return None (i.e., we only PubSub if somebody restarts this)
            for stream_obj in stream_dest_dict.values():
                stream_obj.get(requested_by="send_stream_samples")

            self.add_streams(list(stream_dest_dict.values()), owned=True)
            self.update_streams_in_profile()
            self.subscribe_to_pubsub_owned_streams()
            self.send_profile_to_all()

            # New recorded stream
            self._last_recorded_stream_num += 1

        return True

    def connect_by_role(self, role: str | list[str], filter_fcn: str | None = None,
                        time: float = -1., timeout: float = -1.):
        """Finds and attempts to connect with agents whose profiles match a specific role. It can be optionally
        filtered by a custom function. It returns True if at least one valid agent is found.

        Args:
            role: The role or list of roles to search for.
            filter_fcn: The name of an optional filter function.
            time: The time duration for the action.
            timeout: The action timeout.

        Returns:
            True if at least one agent is found and a connection request is made, False otherwise.
        """
        self.out(f"Asking to get in touch with all agents whose role is {role}")
        assert time is not None and timeout is not None, "Missing basic action information"

        if self.get_action_step() == 0:
            role_list = role if isinstance(role, list) else [role]
            self._found_agents = set()
            at_least_one_is_valid = False

            for role in role_list:
                role = self.ROLE_STR_TO_BITS[role]

                found_addresses1, found_peer_ids1 = self._node_conn.find_addrs_by_role(Agent.ROLE_WORLD_MASTER | role,
                                                                                       return_peer_ids_too=True)
                found_addresses2, found_peer_ids2 = self._node_conn.find_addrs_by_role(Agent.ROLE_WORLD_AGENT | role,
                                                                                       return_peer_ids_too=True)
                found_addresses = found_addresses1 + found_addresses2
                found_peer_ids = found_peer_ids1 + found_peer_ids2

                if filter_fcn is not None:
                    if hasattr(self, filter_fcn):
                        filter_fcn = getattr(self, filter_fcn)
                        if callable(filter_fcn):
                            found_addresses, found_peer_ids = filter_fcn(found_addresses, found_peer_ids)
                    else:
                        self.err(f"Filter function not found: {filter_fcn}")

                self.out(f"Found addresses ({len(found_addresses)}) with role: {role}")
                for f_addr, f_peer_id in zip(found_addresses, found_peer_ids):
                    if not self._node_conn.is_connected(f_peer_id):
                        self.out(f"Asking to get in touch with {f_addr}...")
                        peer_id = self._node_ask_to_get_in_touch_fcn(addresses=f_addr, public=False)
                    else:
                        self.out(f"Not-asking to get in touch with {f_addr}, "
                                 f"since I am already connected to the corresponding peer...")
                        peer_id = f_peer_id
                    if peer_id is not None:
                        at_least_one_is_valid = True
                        self._found_agents.add(peer_id)
                    self.out(f"...returned {peer_id}")
            return at_least_one_is_valid
        else:
            return True

    def find_agents(self, role: str | list[str], engage: bool = False):
        """Locally searches through the agent's known peers (world and public agents) to find agents with a specific
        role. It populates the `_found_agents` set with the peer IDs of matching agents.

        Args:
            role: The role or list of roles to search for.
            engage: If you want to force the found agents to be the ones that you are engaged with.

        Returns:
            True if at least one agent is found, False otherwise.
        """
        self.out(f"Finding an available agent whose role is {role}")
        role_list = role if isinstance(role, list) else [role]
        self._found_agents = set()

        for role_str in role_list:
            agents = self.all_agents
            role_int = self.ROLE_STR_TO_BITS[role_str]
            role_clean = (role_int >> 2) << 2
            for peer_id, profile in agents.items():
                _role_int = self.ROLE_STR_TO_BITS[profile.get_dynamic_profile()['connections']['role']]
                _role_clean = (_role_int >> 2) << 2
                if _role_clean == role_clean:
                    self._found_agents.add(peer_id)  # Peer IDs here

        self.deb(f"[find_agents] Found these agents: {self._found_agents}")
        if engage:
            self._engaged_agents = copy.deepcopy(self._found_agents)
        return len(self._found_agents) > 0

    def next_pref_stream(self):
        """Moves the internal pointer to the next stream in the list of preferred streams, which is often used for
        playlist-like operations. It wraps around to the beginning if it reaches the end.

        Returns:
            True if the move is successful, False if the list is empty.
        """
        if len(self._preferred_streams) == 0:
            self.err(f"Cannot move to the next stream because the list of preferred streams is empty")
            return False

        self._cur_preferred_stream = (self._cur_preferred_stream + 1) % len(self._preferred_streams)
        suffix = ", warning: restarted" if self._cur_preferred_stream == 0 else ""
        self.out(f"Moving to the next preferred stream ({self._preferred_streams[self._cur_preferred_stream]}){suffix}")
        return True

    def first_pref_stream(self):
        """Resets the internal pointer to the first stream in the list of preferred streams. This is useful for
        restarting a playback or processing loop.

        Returns:
            True if the move is successful, False if the list is empty.
        """
        if len(self._preferred_streams) == 0:
            self.err(f"Cannot move to the first stream because the list of preferred streams is empty")
            return False

        self._cur_preferred_stream = 0
        self.out(f"Moving to the first preferred stream ({self._preferred_streams[self._cur_preferred_stream]})")
        return True

    def check_pref_stream(self, what: str = "last"):
        """Checks the position of the current preferred stream within the list. It can check if it's the first, last,
        or if it has completed a full round, among other checks.

        Args:
            what: A string specifying the type of check to perform (e.g., 'first', 'last', 'last_round').

        Returns:
            True if the condition is met, False otherwise.
        """
        valid = ['first', 'last', 'not_first', 'not_last', 'last_round', 'not_last_round', 'last_song', 'not_last_song']
        assert what in valid, f"The what argument can only be one of {valid}"

        self.out(f"Checking if the current preferred playlist item "
                 f"(id: {self._cur_preferred_stream}) is the '{what}' one")
        if what == "first":
            return self._cur_preferred_stream == 0
        elif what == "last":
            return self._cur_preferred_stream == len(self._preferred_streams) - 1
        elif what == "not_first":
            return self._cur_preferred_stream != 0
        elif what == "not_last":
            return self._cur_preferred_stream != len(self._preferred_streams) - 1
        elif what == "last_round":
            return (self._cur_preferred_stream + len(self._preferred_streams) // self._repeat >=
                    len(self._preferred_streams))
        elif what == "not_last_round":
            return (self._cur_preferred_stream + len(self._preferred_streams) // self._repeat <
                    len(self._preferred_streams))
        elif what == "last_song":
            num_streams_in_playlist = len(self._preferred_streams) // self._repeat
            return (self._cur_preferred_stream + 1) % num_streams_in_playlist == 0
        elif what == "not_last_song":
            num_streams_in_playlist = len(self._preferred_streams) // self._repeat
            return (self._cur_preferred_stream + 1) % num_streams_in_playlist != 0

    def set_pref_streams(self, net_hashes: list[str], repeat: int = 1):
        """Fills the agent's list of preferred streams (a playlist). It can repeat the playlist a specified number of
        times and resolves user-provided stream hashes to their full network hashes.

        Args:
            net_hashes: A list of stream hashes to add to the playlist.
            repeat: The number of times to repeat the playlist.

        Returns:
            Always True.
        """
        self.out(f"Setting up a list of {len(net_hashes)} preferred streams")
        self._cur_preferred_stream = 0
        self._preferred_streams = []
        self._repeat = repeat
        for i in range(0, self._repeat):
            for net_hash in net_hashes:

                # We are tolerating both peer_id:name_or_group and also peer_id::ps:name_or_group
                components = net_hash.split(":")
                peer_id = components[0]
                name_or_group = components[-1]
                net_hash_to_streams = self.find_streams(peer_id=peer_id, name_or_group=name_or_group)
                for _net_hash in net_hash_to_streams.keys():
                    self._preferred_streams.append(_net_hash)

        return True

    def evaluate(self, stream_hash: str, how: str, steps: int = 100, re_offset: bool = False):
        """Evaluates the performance of agents that have completed a generation task. It compares the generated data
        from each agent with a local stream (which can be a ground truth or reference stream) using a specified
        comparison method.

        Args:
            stream_hash: The hash of the local stream to use for comparison.
            how: The name of the comparison method to use.
            steps: The number of steps to perform the evaluation.
            re_offset: A boolean to indicate whether to re-offset the streams.

        Returns:
            True if the evaluation is successful, False otherwise.
        """
        if not self.buffer_generated_by_others:
            self.err("Cannot evaluate if not buffering data generated by others")
            return False

        if stream_hash == "<playlist>":
            net_hash = self._preferred_streams[self._cur_preferred_stream]
        else:
            net_hash = self.user_stream_hash_to_net_hash(stream_hash)

        self._eval_results = {}
        self.deb(f"[eval] Agents returning streams: {self._agents_who_completed_what_they_were_asked}")
        for peer_id in self._agents_who_completed_what_they_were_asked:
            received_net_hash = self.last_buffered_peer_id_to_info[peer_id]["net_hash"]
            self.out(f"Comparing {net_hash} with {received_net_hash}")
            eval_result, ret = self.__compare_streams(net_hash_a=net_hash,
                                                      net_hash_b=received_net_hash,
                                                      how=how, steps=steps, re_offset=re_offset)
            self.out(f"Result of the comparison: {eval_result}")
            if not ret:
                return False
            else:
                peer_id = DataProps.peer_id_from_net_hash(received_net_hash)
                self._eval_results[peer_id] = eval_result

        return True

    def compare_eval(self, cmp: str, thres: float, good_if_true: bool = True):
        """Compares the results of a previous evaluation to a given threshold or finds the best result among all
        agents. It can check for minimum, maximum, or simple threshold-based comparisons, and it populates a list of
        'valid' agents that passed the comparison.

        Args:
            cmp: The comparison operator (e.g., '<', '>', 'min').
            thres: The threshold value for comparison.
            good_if_true: A boolean to invert the pass/fail logic.

        Returns:
            True if at least one agent passed the comparison, False otherwise.
        """
        assert cmp in ["<", ">", ">=", "<=", "min", "max"], f"Invalid comparison operator: {cmp}"
        assert thres >= 0. or cmp in ["min", "max"], f"Invalid evaluation threshold: {thres} (it must be in >= 0.)"

        self._valid_cmp_agents = set()
        msgs = []
        best_so_far = -1

        min_or_max = None
        leq_or_geq = None
        if cmp in ["min", "max"]:
            min_or_max = "minimum" if cmp == "min" else "maximum"
            leq_or_geq = "<=" if cmp == "min" else ">="

        for agent, eval_result in self._eval_results.items():
            if cmp not in ["min", "max"]:
                self.out(f"Checking if result {eval_result} {cmp} {thres}, for agent {agent}")
            else:
                if thres >= 0:
                    self.out(f"Checking if result {eval_result} is the {min_or_max} so far, "
                             f"only if {leq_or_geq} {thres}, for agent {agent}")
                else:
                    self.out(f"Checking if result {eval_result} is the {min_or_max} so far, for agent {agent}")

            if eval_result < 0.:
                self.err(f"Invalid evaluation result: {eval_result}")
                return False

            if cmp != "min" and cmp != "max":
                outcome = False
                if cmp == "<" and eval_result < thres:
                    outcome = True
                elif cmp == "<=" and eval_result <= thres:
                    outcome = True
                elif cmp == ">" and eval_result > thres:
                    outcome = True
                elif cmp == ">=" and eval_result >= thres:
                    outcome = True

                if cmp[0] == "<" or cmp[0] == "<=":
                    alias = 'error level' if good_if_true else 'mark'
                else:
                    alias = 'mark' if good_if_true else 'error level'

                if good_if_true:
                    if outcome:
                        msgs.append(f"Agent {agent} passed with {alias} {eval_result}/{thres}")
                        self._valid_cmp_agents.add(agent)
                    else:
                        msgs.append(f"Agent {agent} did not pass")
                else:
                    if outcome:
                        msgs.append(f"Agent {agent} did not pass")
                    else:
                        msgs.append(f"Agent {agent} passed with {alias} {eval_result}/{thres}")
                        self._valid_cmp_agents.add(agent)

                if len(msgs) > 1:
                    msgs[-1] = str(msgs[-1].lower())[0] + msgs[-1][1:]
            else:
                if ((cmp == "min" and (thres < 0 or eval_result <= thres) and
                     (eval_result < best_so_far or best_so_far < 0)) or
                        (cmp == "max" and (thres < 0 or eval_result >= thres) and
                         (eval_result > best_so_far or best_so_far < 0))):
                    best_so_far = eval_result
                    self._valid_cmp_agents = {agent}
                    msgs = [f"The best agent is {agent}"]
                else:
                    msgs = [f"No best agent found for the considered threshold ({thres})"]

        if len(self._valid_cmp_agents) == 0:

            # # cheating (hack):
            # self._valid_cmp_agents.append(agent)
            # self.out(", ".join(msgs))
            # return True
            self.err(f"The evaluation was not passed by any agents")
            return False
        else:
            self.out(", ".join(msgs))
            return True

    def suggest_role_to_world(self, agent: str | None, role: str):
        """Suggests a role change for one or more agents to the world master. It iterates through the involved agents,
        checks if their current role differs from the suggested one, and sends a role suggestion message to the
        world master.

        Args:
            agent: The ID of the agent or a wildcard to suggest the role for.
            role: The new role to suggest (as a string).

        Returns:
            True if the suggestion was sent successfully, False otherwise.
        """
        self.out("Suggesting role to world")

        agents = self.__involved_agents(agent)
        role_bits = (self.ROLE_STR_TO_BITS[role] >> 2) << 2

        content = []

        for _agent in agents:
            cur_role_bits = self.ROLE_STR_TO_BITS[self.all_agents[_agent].get_dynamic_profile()['connections']['role']]
            cur_role_bits = (cur_role_bits >> 2) << 2
            if cur_role_bits == role_bits:
                self.out(f"Not suggesting to change the role of {_agent} "
                         f"since it has already such a role")
            else:
                self.out(f"Suggesting to change the role of {_agent} to {self.ROLE_BITS_TO_STR[role_bits]}")
                content.append({'peer_id': _agent, 'role': role_bits})

        if len(content) > 0:
            world_peer_id = self._node_conn.get_world_peer_id()
            if not self._node_conn.send(world_peer_id, channel_trail=None,
                                        content=content,
                                        content_type=Msg.ROLE_SUGGESTION):
                self.err("Failed to send role suggestion to the world")
                return False
        return True

    def suggest_badges_to_world(self, agent: str | None = None,
                                score: float = -1.0, badge_type: str = "completed",
                                badge_description: str | None = None):
        """Suggests one or more badges to the world master for specific agents. This is typically used to reward agents
        for completing tasks, such as for a competition. It sends a message with the badge details, including the score
        and type, to the world master.

        Args:
            agent: The ID of the agent or a wildcard for which to suggest the badge.
            score: The score associated with the badge.
            badge_type: The type of badge (e.g., 'completed').
            badge_description: An optional description for the badge.

        Returns:
            True if the badge suggestion was sent successfully, False otherwise.
        """
        self.out("Suggesting one or more badges to world")

        if score < 0.:
            self.err("Invalid score (did you specify the 'score' argument? it must be positive)")
            return False

        agents = self.__involved_agents(agent)
        world_peer_id = self._node_conn.get_world_peer_id()

        if badge_type not in Agent.BADGE_TYPES:
            self.err(f"Unknown badge type: {badge_type}")
            return False

        list_of_badge_dictionaries = []
        for peer_id in agents:
            list_of_badge_dictionaries.append({'peer_id': peer_id,
                                               'score': score,
                                               'badge_type': badge_type,
                                               'badge_description': badge_description,
                                               'agent_token': self._node_conn.get_last_token(peer_id)})

        if not self._node_conn.send(world_peer_id, channel_trail=None,
                                    content=list_of_badge_dictionaries,
                                    content_type=Msg.BADGE_SUGGESTIONS):
            self.err("Failed to send badge suggestions to the world")
            return False
        else:
            return True

    def __ask_gen_or_learn(self, for_what: str, agent: str,
                           u_hashes: list[str] | None,
                           yhat_hashes: list[str] | None,
                           samples: int = 100, time: float = -1., timeout: float = -1., ref_uuid: str | None = None):
        """A private helper method that encapsulates the logic for sending a 'do_gen' or 'do_learn' action request to
        another agent. It handles the normalization of stream hashes, sets up recipients for direct messages, and adds
        the target agent to the list of agents asked.

        Args:
            for_what: A string indicating whether to ask for 'gen' or 'learn'.
            agent: The ID of the agent to send the request to.
            u_hashes: A list of input stream hashes.
            yhat_hashes: A list of target stream hashes (for learning).
            samples: The number of samples.
            time: The time duration.
            timeout: The request timeout.
            ref_uuid: The UUID for the request.

        Returns:
            True if the request was sent successfully, False otherwise.
        """
        if agent not in self.all_agents:
            self.err(f"Unknown agent: {agent}")
            return False

        assert for_what in ["gen", "learn"]

        if for_what == "learn":
            for yhat_hash in yhat_hashes:
                yhat_stream_dict = self.known_streams[yhat_hash]
                for yhat_stream in yhat_stream_dict.values():
                    if isinstance(yhat_stream, BufferedDataStream):
                        y_text = yhat_stream.to_text_snippet(length=200)
                        if y_text is not None and len(y_text) > 0:
                            self.out("Asking to learn: \"" + y_text + "\"")

        # Setting recipient in the case of direct messages
        # (differently, in case of pubsub, the agent is already sending messages to all)
        if u_hashes is not None:
            for u_hash in u_hashes:
                if not DataProps.is_pubsub_from_net_hash(u_hash):
                    self.recipients[u_hash] = agent
        if yhat_hashes is not None:
            for yhat_hash in yhat_hashes:
                if not DataProps.is_pubsub_from_net_hash(yhat_hash):
                    self.recipients[yhat_hash] = agent

        # Triggering
        if for_what == "gen":
            if self.set_next_action(agent, action="do_gen", args={"u_hashes": u_hashes,
                                                                  "samples": samples, "time": time,
                                                                  "timeout": timeout},
                                    ref_uuid=ref_uuid):
                self._agents_who_were_asked.add(agent)
                return True
            else:
                self.err(f"Unable to ask {agent} to generate")
                return False
        elif for_what == "learn":
            if self.set_next_action(agent, action="do_learn", args={"u_hashes": u_hashes, "yhat_hashes": yhat_hashes,
                                                                    "samples": samples, "time": time,
                                                                    "timeout": timeout},
                                    ref_uuid=ref_uuid):
                self._agents_who_were_asked.add(agent)
                return True
            else:
                self.err(f"Unable to ask {agent} to learn to generate")
                return False

    def __process_streams(self,
                          u_hashes: list[str] | None,
                          yhat_hashes: list[str] | None,
                          learn: bool = False,
                          recipient: str | None = None,
                          ref_uuid: str | None = None):
        """A private helper method that contains the core logic for processing data streams, either for generation or
        learning. It reads input streams, passes them to the agent's processor, and handles the output streams.
        It's designed to be called repeatedly by multistep actions like `do_gen` and `do_learn`.

        Args:
            u_hashes: A list of input stream hashes.
            yhat_hashes: A list of target stream hashes (for learning).
            learn: A boolean to indicate if the task is a learning task.
            recipient: The ID of the agent to send data back to.
            ref_uuid: The UUID for the request.

        Returns:
            True if the stream processing is successful, False otherwise.
        """

        # Getting current step index
        k = self.get_action_step()

        # Checking data and creating new buffered streams
        if k == 0:
            self.deb("[__process_streams] First action step")

            # Checking data
            if u_hashes is not None:
                for u_hash in u_hashes:
                    if u_hash is not None and u_hash not in self.known_streams:
                        self.err(f"Unknown stream (u_hash): {u_hash}")
                        return False
            if yhat_hashes is not None:
                for yhat_hash in yhat_hashes:
                    if yhat_hash is not None and yhat_hash not in self.known_streams:
                        self.err(f"Unknown stream (yhat_hash): {yhat_hash}")
                        return False

        if self.is_last_action_step():
            self.deb("[__process_streams] Last action step detected")

        self.deb(f"[__process_streams] Generating data, step {k}")

        # Generate output
        outputs, data_tag_from_inputs = (
            self.generate(input_net_hashes=u_hashes, first=(k == 0), last=self.is_last_action_step(),
                          ref_uuid=ref_uuid))
        if outputs is None:
            return False
        self.deb(f"[__process_streams] data_tag_from_inputs: {data_tag_from_inputs}")
        if data_tag_from_inputs is None:
            data_tag_from_inputs = -1
            self.deb(f"[__process_streams] data_tag_from_inputs (forced): {data_tag_from_inputs}")

        # Learn
        if learn:
            self.deb(f"[__process_streams] learning, step {k}")
            loss_values, data_tags_from_targets = self.learn_generate(outputs=outputs, targets_net_hashes=yhat_hashes)
            self.deb(f"[__process_streams] data_tags_from_targets: {data_tags_from_targets}")

            if loss_values is None:
                return False
            else:
                self.out(f"Losses: {loss_values}")

            # Fusing data tags
            data_tags = [data_tag_from_inputs if _data_tag == -1 else _data_tag for _data_tag in data_tags_from_targets]
        else:
            data_tags = [data_tag_from_inputs] * len(outputs)
        self.deb(f"[__process_streams] data_tags (final): {data_tags}")

        # Set each data sample in "outputs" to the right stream
        i = 0
        for net_hash, stream_dict in self.proc_streams.items():

            # Setting the data sample
            for name, stream_obj in stream_dict.items():

                # Public output streams are only considered if the agent IS NOT acting in a world
                # private output streams are only considered if the agent IS acting in a world
                if self.behaving_in_world() != stream_obj.props.is_public():

                    # Guessing recipient of the communication
                    if i == 0:
                        self.recipients[net_hash] = recipient \
                            if not DataProps.is_pubsub_from_net_hash(net_hash) else None

                    self.deb(f"[__process_streams] Setting the {i}-th network output to stream with "
                             f"net_hash: {net_hash}, name: {name}")

                    # Here we exploit the fact that streams were inserted in order
                    try:
                        stream_obj.set(stream_obj.props.check_and_postprocess(outputs[i]), data_tags[i])
                    except Exception as e:
                        self.err(f"Error while post-processing the processor output\nException: {e}")
                        return False

                    if k == 0:
                        stream_obj.set_uuid(ref_uuid, expected=False)
                        stream_obj.set_uuid(None, expected=True)
                    i += 1

        return True

    def __complete_do(self, do_what: str, peer_id_who_asked: str, all_hashes: list[str] | None,
                      send_back_confirmation: bool = True):
        """A private helper method to be called at the end of a `do_gen` or `do_learn` action. It performs cleanup
        tasks, such as clearing UUIDs on streams, and sends a confirmation message back to the requesting agent.

        Args:
            do_what: A string ('gen' or 'learn') indicating which task was completed.
            peer_id_who_asked: The ID of the agent who requested the task.
            all_hashes: A list of all stream hashes involved in the task.
            send_back_confirmation: A boolean to indicate if a confirmation message should be sent.

        Returns:
            True if the completion process is successful, False otherwise.
        """
        assert do_what in ["gen", "learn"]

        if do_what == "gen":
            for net_hash, stream_dict in self.proc_streams.items():
                for stream in stream_dict.values():
                    if isinstance(stream, BufferedDataStream):
                        y_text = stream.to_text_snippet(length=200)
                        if y_text is not None:
                            self.out("Generated: \"" + y_text + "\"")

        for stream_dict in self.proc_streams.values():
            for stream_obj in stream_dict.values():
                if stream_obj.props.is_public() != self.behaving_in_world():
                    stream_obj.mark_uuid_as_clearable()

        if all_hashes is not None:
            for net_hash in all_hashes:
                for stream_obj in self.known_streams[net_hash].values():
                    stream_obj.set_uuid(None, expected=False)
                    stream_obj.set_uuid(None, expected=True)

        # Confirming
        if send_back_confirmation:
            if self.set_next_action(peer_id_who_asked, action="done_" + do_what, args={}):
                return True
            else:
                self.err(f"Unable to confirm '{do_what}' to {peer_id_who_asked}")
                return False
        else:
            return True

    def __compare_streams(self, net_hash_a: str, net_hash_b: str,
                          how: str = "mse", steps: int = 100, re_offset: bool = False):
        """A private helper method that compares two buffered data streams based on a specified metric (e.g., MSE,
        max accuracy). It handles stream compatibility checks, data retrieval, and the actual comparison, returning a
        dissimilarity score.

        Args:
            net_hash_a: The network hash of the first stream.
            net_hash_b: The network hash of the second stream.
            how: The comparison metric ('mse', 'max', 'geqX').
            steps: The number of samples to compare.
            re_offset: A boolean to re-align stream tags before comparison.

        Returns:
            A tuple containing the dissimilarity score and a success flag (e.g., `(0.5, True)`).
        """
        if net_hash_a not in self.known_streams:
            self.err(f"Unknown stream (net_hash_a): {net_hash_a}")
            return -1., False

        if net_hash_b not in self.known_streams:
            self.err(f"Unknown stream (net_hash_b): {net_hash_b}")
            return -1., False

        if steps <= 0:
            self.err(f"Invalid number of steps: {steps}")
            return -1., False

        if how not in ["mse", "max"] and not how.startswith("geq"):
            self.err(f"Data can be compared by MSE, or by comparing the argmax ('max'), or comparing the number "
                     f"of corresponding bits (obtained by 'geqX', where 'X' is a number). Unknown: {how})")
            return -1., False

        stream_dict_a = self.known_streams[net_hash_a]
        stream_dict_b = self.known_streams[net_hash_b]

        if len(stream_dict_a) == 1 and len(stream_dict_b) == 1:

            # If there is only 1 stream is each group, things are easy
            stream_a = next(iter(stream_dict_a.values()))
            stream_b = next(iter(stream_dict_b.values()))
        elif len(stream_dict_a) == 1 and len(stream_dict_b) > 1:

            # If there is only 1 stream is one of the groups, we look for a compatible stream in the other group,
            # giving priority to streams with labels
            stream_a = next(iter(stream_dict_a.values()))
            stream_b = None
            for stream_obj in stream_dict_b.values():
                if (stream_a.get_props().has_tensor_labels() and stream_obj.get_props().has_tensor_labels() and
                        stream_obj.get_props().is_compatible(stream_a.get_props())):
                    stream_b = stream_obj
                    break
            if stream_b is None:
                for stream_obj in stream_dict_b.values():
                    if stream_obj.get_props().is_compatible(stream_a.get_props()):
                        stream_b = stream_obj
                        break
        elif len(stream_dict_a) > 1 and len(stream_dict_b) == 1:

            # If there is only 1 stream is one of the groups, we look for a compatible stream in the other group,
            # giving priority to streams with labels
            stream_a = None
            stream_b = next(iter(stream_dict_b.values()))
            for stream_obj in stream_dict_a.values():
                if (stream_b.get_props().has_tensor_labels() and stream_obj.get_props().has_tensor_labels() and
                        stream_obj.get_props().is_compatible(stream_b.get_props())):
                    stream_a = stream_obj
                    break
            if stream_a is None:
                for stream_obj in stream_dict_a.values():
                    if stream_obj.get_props().is_compatible(stream_b.get_props()):
                        stream_a = stream_obj
                        break
        else:

            # If both groups have more than a stream, let's give priority to streams with labels to find a match
            stream_a = None
            stream_b = None
            for stream_obj_a in stream_dict_a.values():
                if not stream_obj_a.get_props().has_tensor_labels():
                    continue
                if stream_a is not None and stream_b is not None:
                    break
                for stream_obj_b in stream_dict_b.values():
                    if (stream_obj_b.get_props().has_tensor_labels() and
                            stream_obj_a.get_props().is_compatible(stream_obj_b.get_props())):
                        stream_a = stream_obj_a
                        stream_b = stream_obj_b
                        break
            if stream_a is None and stream_b is None:
                for stream_obj_a in stream_dict_a.values():
                    if stream_a is not None and stream_b is not None:
                        break
                    for stream_obj_b in stream_dict_b.values():
                        if stream_obj_a.get_props().is_compatible(stream_obj_b.get_props()):
                            stream_a = stream_obj_a
                            stream_b = stream_obj_b
                            break

        if stream_a is None:
            self.err(f"Cannot find the data stream to consider in the comparison, {net_hash_a}")
            return -1., False
        if stream_b is None:
            self.err(f"Cannot find the data stream to consider in the comparison, {net_hash_b}")
            return -1., False

        if not isinstance(stream_a, BufferedDataStream):
            self.err(f"Can only compare buffered streams and {net_hash_a} is not buffered")
            return -1., False

        if not isinstance(stream_b, BufferedDataStream):
            self.err(f"Can only compare buffered streams and {net_hash_b} is not buffered")
            return -1., False

        if steps > len(stream_a) and steps > len(stream_b):
            self.err(f"Cannot compare streams for {steps} steps, since both of them are shorter "
                     f"(length of the first stream is {len(stream_a)}, of the second stream is {len(stream_b)})")

        if not stream_a.get_props().is_compatible(stream_b.get_props()):
            self.err(f"Cannot compare incompatible streams")

        stream_a.restart()
        stream_b.restart()

        def compare(_a: torch.Tensor | str, _b: torch.Tensor | str, _how: str = "mse") -> float:
            """Compare two samples of signals or descriptors, returning a dissimilarity score >= 0."""

            assert how in ['mse', 'max', 'same'] or how.startswith("geq"), f"Invalid comparison in terms of {how}"

            if isinstance(_a, torch.Tensor) and isinstance(_b, torch.Tensor):
                if _a.dtype == torch.long and _b.dtype == torch.long:  # Token IDS
                    return 1. - float((_a == _b).sum().item()) / a.numel()  # Accuracy
                elif how == "mse":
                    ret = torch.nn.functional.mse_loss(_a, _b, reduction='mean')
                elif how == "max":
                    ret = 1. - float((torch.argmax(_a) == torch.argmax(_b)).sum().item()) / a.numel()
                elif how == "same":
                    ret = 1. - float(torch.eq(_a, _b).sum()) / a.numel()
                else:
                    thres = float(how[3:])
                    ret = 1. - float(torch.sum((_a > thres) == (_b > thres)).item()) / a.numel()
            else:
                ret = 1. - float(_a == _b)  # Strings (always handled as 'same')
            return ret

        # Comparing data (averaging)
        o = 0.
        k_b = 0
        a_tag_offset = 0
        b_tag_offset = 0
        a_tag = None
        a_tag_prev = None
        for k_a in range(0, steps):

            restart_detected = False
            if a_tag is not None:
                a_tag_prev = a_tag

            # Signals or descriptors
            a, a_tag = stream_a[k_a]
            b, b_tag = stream_b[k_b]

            # If the streams do not share the same first tag equal to zero, and we asked to re-offset them,
            # then we force the initial offsets to be zero on both
            # if not, then re-offset the tags
            if k_a == 0 and k_b == 0 and re_offset:
                a_tag_offset = a_tag
                b_tag_offset = b_tag

            # Offset-based tags
            a_tag_w_offset = a_tag - a_tag_offset
            b_tag_w_offset = b_tag - b_tag_offset

            # Checking
            if a is None:
                self.err("Cannot compare stream samples if the reference stream yields None")
                return -1., False

            # Some streams might have been pre-buffered in advance, and have increasing data tags belonging to finite,
            # fixed set (such as 0, 1, 2, ..., N). when continuously streaming them, we will go from tag N to tag 0 at
            # a certain point, which is a "restart".
            # We have to remember that this happened, and we do it for stream "a", our "reference" stream.
            # Then, below, we will fix tags on stream "b" if needed, considering that such a restart happened.
            if a_tag_prev is not None and a_tag < a_tag_prev:
                restart_detected = True

            # Some streams might have been pre-buffered in advance, and have a fixed data tag (usually -1).
            # Being it negative, it will happen that the data tag will be replaced by a clock cycle, but this function
            # does not change clock cycles at all, so all samples will have the exact same data tag.
            # The following code automatically advances the tag by 1 for stream "a", that is expected to be the
            # reference stream (i.e., the one for which the agent has all samples, with no missing data in between)
            if a_tag_prev is not None and a_tag <= a_tag_prev:
                a_tag = a_tag_prev + 1  # Fixed tag detected (patching)
                a_tag_w_offset = a_tag - a_tag_offset

            # Fixing
            if b is None:
                o = o + (1. if how != "mse" else (o / steps) * 1.1)
                self.deb(f"[__compare_streams] The second stream yields None")
            else:
                if b_tag_w_offset == a_tag_w_offset:
                    o += compare(a, b, how)
                    k_b += 1
                    self.deb(f"[__compare_streams] Comparing tags: {a_tag} vs {b_tag} "
                             f"(with offsets: {a_tag_w_offset} vs {b_tag_w_offset}), samples: {a} vs {b}")
                elif b_tag_w_offset > a_tag_w_offset:
                    if not restart_detected:
                        o = o + (1. if how != "mse" else (o / steps) * 1.1)  # Don't change k_b, some samples missing
                        self.deb(f"[__compare_streams] (b) Comparing tags: {a_tag} vs {b_tag} -> "
                                 f"expected one was missing "
                                 f"(with offsets: {a_tag_w_offset} vs {b_tag_w_offset}) "
                                 f"samples: {a} vs {b}")
                    else:
                        o = o + (1. if how != "mse" else (o / steps) * 1.1)
                        self.deb(f"[__compare_streams] (c) Comparing tags: {a_tag} vs {b_tag} -> "
                                 f"expected one was missing "
                                 f"(with offsets: {a_tag_w_offset} vs {b_tag_w_offset}) "
                                 f"samples: {a} vs {b}")
                        k_b += 1  # A restart was detected, it means that "stream_b" is behind, let's move it ahead
                elif b_tag_w_offset < a_tag_w_offset:
                    self.deb(f"[__compare_streams] (d) Comparing tags: {a_tag} vs {b_tag} -> too early w.r.t. expected "
                             f"(with offsets: {a_tag_w_offset} vs {b_tag_w_offset}) "
                             f"samples: {a} vs {b}")
                    return -1., False

        self.deb(f"[__compare_streams] Error: {o / steps}")

        # Input("*** press enter to continue ***")
        return o / steps, True

    def __involved_agents(self, agent: str | None):
        """A private helper method that resolves an agent ID or a wildcard into a list of specific peer IDs.
        It can resolve a single agent, a group of agents that passed a previous comparison (`<valid_cmp>`), or all
        currently engaged agents.

        Args:
            agent: The agent ID or wildcard string.

        Returns:
            A list of peer IDs corresponding to the involved agents.
        """
        peer_id = agent
        involved_agents = [peer_id] if peer_id is not None and peer_id != "<valid_cmp>" else (
            self._valid_cmp_agents) if peer_id is not None and peer_id == "<valid_cmp>" else self._engaged_agents
        if len(involved_agents) == 0:
            self.err("Not engaged to any agents or no agent specified")
        return involved_agents
