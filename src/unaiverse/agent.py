"""
 █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
 ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
 ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
 ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
 ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
 ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
  ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 

~ Registration/Login: https://unaiverse.io
~ Code Repositories:  https://github.com/collectionlessai/
~ Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi

A Collectionless AI Project (https://collectionless.ai)
"""
import copy
import json
import uuid
import torch
from unaiverse.dataprops import DataProps
from unaiverse.agent_basics import AgentBasics
from unaiverse.networking.p2p.messages import Msg
from unaiverse.streams import BufferedDataStream, DataStream


class Agent(AgentBasics):
    """This class contains those basic actions that can be performed by every agent."""

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
        my_role = self.ROLE_STR_TO_BITS[self._node_profile.get_dynamic_profile()['connections']['role']]
        for found_agent in self._found_agents:
            if self.set_next_action(found_agent, action="get_engagement",
                                    args={"sender_role": self.ROLE_BITS_TO_STR[my_role]}):
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

        # confirming
        if self.available:
            acceptable_role_int = self.ROLE_STR_TO_BITS[acceptable_role]
            if "~" not in acceptable_role:
                sender_role_int = (self.ROLE_STR_TO_BITS[sender_role] >> 2) << 2
            else:
                sender_role_int = self.ROLE_STR_TO_BITS[sender_role]

            if acceptable_role_int == sender_role_int:
                if self.set_next_action(_requester, "got_engagement"):
                    self._engaged_agents.add(_requester)

                    # marking this agent as not available since it engaged with another one
                    self.available = False
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

            # marking this agent as not available since it engaged with another one
            self.available = False

            # removing the engaged agent from the list of found agents, to avoid sending him another engagement request
            self._found_agents.discard(_requester)
            return True
        else:
            self.err(f"Unable to confirm engagement with {_requester}")
            return False

    def send_disengagement(self, send_disconnection_too: bool = False):
        """Ask for disengagement.

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

        self._engaged_agents.discard(_requester)   # remove if present

        # marking this agent as available if not engaged to any agent
        self.available = len(self._engaged_agents) == 0
        return True

    def disengage_all(self):
        """Disengage all the previously engaged agents.

        Returns:
            True if the disengagement procedure was successfully executed, False otherwise.
        """
        self.out(f"Disengaging all agents")
        self._engaged_agents = set()

        # marking this agent as available
        self.available = True
        return True

    def disconnect_by_role(self, role: str | list[str]):
        self.out(f"Disconnecting agents with role: {role}")
        if self.find_agents(role):
            found_agents = copy.deepcopy(self._found_agents)
            for agent in found_agents:
                self._node_purge_fcn(agent)  # this will also call remove_agent, that will call remove_streams
        return True

    def disconnected(self, agent: str | None = None):

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.out(f"Checking if all these agents are not connected to me anymore: {involved_agents}")
        somebody_still_connected = False
        for agent in involved_agents:
            if agent in self.world_agents or agent in self.public_agents:
                somebody_still_connected = True
                break
        return somebody_still_connected

    def received_some_asked_data(self, processing_fcn: str | None = None):
        """Checks if at least one of the agents who were asked for something (generate, learn, ...) sent some data."""

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

        # create a copy of the input hashes, normalizing them in the appropriate way
        u_hashes_copy: list[str | None] = [None] * len(u_hashes)
        for i in range(len(u_hashes_copy)):
            if u_hashes_copy[i] == "<playlist>":
                # from <playlist> to the current element of the playlist
                u_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:
                # from a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                u_hashes_copy[i] = self.user_stream_hash_to_net_hash(u_hashes[i])

        # generate a new UUID for this request
        ref_uuid = uuid.uuid4().hex[0:8] if ask_uuid is None else ask_uuid
        if ignore_uuid:
            ref_uuid = None

        # if the input streams are all owned by this agent, discard UUID
        all_owned = True
        for i in range(len(u_hashes_copy)):
            if u_hashes_copy[i] not in self.owned_streams:
                all_owned = False
                break
        if not all_owned:
            ref_uuid = None

        for i in range(len(u_hashes_copy)):

            # if there are our own streams involved, and they are buffered, let's plan to restart them when we will
            # start sending them through the net: moreover, let's set the local stream UUID appropriately to
            # the generated UUID
            if u_hashes_copy[i] in self.owned_streams:
                stream_dict = self.known_streams[u_hashes_copy[i]]
                for stream_name, stream_obj in stream_dict.items():

                    # plan to restart buffered streams
                    if isinstance(stream_obj, BufferedDataStream):
                        stream_obj.plan_restart_before_next_get(requested_by="send_stream_samples")

                    # activate the stream (if it was off)
                    stream_obj.enable()

                    # set UUID to the generated one
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

        # preparing the buffered stream where to store data, if needed
        if len(correctly_asked) > 0:

            # saving
            self._last_ref_uuid = ref_uuid

            # for each agent that we involve in this request....
            for peer_id in correctly_asked:

                # finding the streams generated by the processor of the agent we asked to generate
                processor_streams = self.find_streams(peer_id, name_or_group="processor")

                # for each stream generated by the processor of the agent we asked to generate...
                for net_hash, stream_dict in processor_streams.items():

                    # set the appropriate UUID to the one we created in this method
                    for stream in stream_dict.values():
                        stream.set_uuid(None, expected=False)
                        stream.set_uuid(ref_uuid, expected=True)  # setting the "expected" one

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
        else:
            self.err("Unknown requester (None)")
            return False

        # check what is the step ID of the multistep action
        k = self.get_action_step()

        # in the first step of this action, we change the UUID of the local stream associated to the input data we will
        # use to handle this action, setting expectations to avoid handling tags of old data
        if k == 0:

            # warning: we are not normalizing the hashes, we should do it if this action is called directly
            if u_hashes is not None:
                for net_hash in u_hashes:
                    if net_hash in self.known_streams:
                        for stream_obj in self.known_streams[net_hash].values():

                            # if the data arrived before this action, then the UUID is already set, and here there is
                            # no need to do anything; if the data has not yet arrived (common case) ...
                            if stream_obj.get_uuid(expected=False) != _request_uuid:
                                stream_obj.set_uuid(None, expected=False)  # clearing UUID
                                stream_obj.set_uuid(_request_uuid, expected=True)  # setting expectations
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

        # searching for the processor-streams of the agent who generated data
        processor_streams = self.find_streams(_requester, name_or_group="processor")
        if processor_streams is None or len(processor_streams) == 0:
            self.err("Unexpected confirmation of finished generation")
            return False

        # remembering that the agent that invoked this action is the one who generated the data, and what he generated
        # could be used in future action (for example, in evaluation processes)
        self._agents_who_completed_what_they_were_asked.add(_requester)

        # clearing the UUID of the local streams associated to the agent who generated
        for net_hash, stream_dict in processor_streams.items():
            for stream_obj in stream_dict.values():
                stream_obj.set_uuid(None, expected=False)
                stream_obj.set_uuid(None, expected=True)

        # if one or more of my streams where used as arguments of the generation request I did (ask_gen), then their
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

        # create a copy of the input hashes, normalizing them in the appropriate way
        u_hashes_copy = [x for x in u_hashes]
        for i in range(len(u_hashes_copy)):
            if u_hashes_copy[i] == "<playlist>":
                # from <playlist> to the current element of the playlist
                u_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:
                # from a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                u_hashes_copy[i] = self.user_stream_hash_to_net_hash(u_hashes_copy[i])

        # create a copy of the target hashes, normalizing them in the appropriate way
        yhat_hashes_copy = [x for x in yhat_hashes]
        for i in range(len(yhat_hashes_copy)):
            if yhat_hashes_copy[i] == "<playlist>":
                # from <playlist> to the current element of the playlist
                yhat_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:
                # from a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                yhat_hashes_copy[i] = self.user_stream_hash_to_net_hash(yhat_hashes_copy[i])

        # generate a new UUID for this request
        ref_uuid = uuid.uuid4().hex[0:8] if ask_uuid is None else ask_uuid
        if ignore_uuid:
            ref_uuid = None

        # if the input streams are all owned by this agent, discard UUID
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

            # if there are our own streams involved, and they are buffered, let's plan to restart them when we will
            # start sending them through the net: moreover, let's set the local stream UUID appropriately to
            # the generated UUID
            if u_hashes_copy[i] in self.owned_streams:
                stream_dict = self.known_streams[u_hashes_copy[i]]
                for stream_name, stream_obj in stream_dict.items():

                    # plan to restart buffered streams
                    if isinstance(stream_obj, BufferedDataStream):
                        stream_obj.plan_restart_before_next_get(requested_by="send_stream_samples")

                    # activate the stream (if it was off)
                    stream_obj.enable()

                    # set UUID to the generated one
                    stream_obj.set_uuid(ref_uuid=ref_uuid, expected=False)
                    stream_obj.set_uuid(ref_uuid=None, expected=True)

        for i in range(len(yhat_hashes_copy)):

            # if there are our own streams involved, and they are buffered, let's plan to restart them when we will
            # start sending them through the net: moreover, let's set the local stream UUID appropriately to
            # the generated UUID
            if yhat_hashes_copy[i] in self.owned_streams:
                stream_dict = self.known_streams[yhat_hashes_copy[i]]
                for stream_name, stream_obj in stream_dict.items():

                    # plan to restart buffered streams
                    if isinstance(stream_obj, BufferedDataStream):
                        stream_obj.plan_restart_before_next_get(requested_by="send_stream_samples")

                    # activate the stream (if it was off)
                    stream_obj.enable()

                    # set UUID to the generated one
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

        # preparing the buffered stream where to store data, if needed
        if len(correctly_asked) > 0:

            # for each agent that we involve in this request....
            for peer_id in correctly_asked:

                # finding the streams generated by the processor of the agent we asked to generate
                processor_streams = self.find_streams(peer_id, name_or_group="processor")

                # for each stream generated by the processor of the agent we asked to generate...
                for net_hash, stream_dict in processor_streams.items():

                    # set the appropriate UUID to the one we created in this method
                    for stream in stream_dict.values():
                        stream.set_uuid(None, expected=False)
                        stream.set_uuid(ref_uuid, expected=True)  # setting the "expected" one

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

        # check what is the step ID of the multistep action
        k = self.get_action_step()

        # in the first step of this action, we change the UUID of the local stream associated to the input data we will
        # use to handle this action, setting expectations to avoid handling tags of old data
        if k == 0:

            # warning: we are not normalizing the hashes, we should do it if this action is called directly
            if u_hashes is not None:
                for net_hash in u_hashes:
                    if net_hash in self.known_streams:
                        for stream_obj in self.known_streams[net_hash].values():

                            # if the data arrived before this action, then the UUID is already set, and here there is
                            # no need to do anything; if the data has not yet arrived (common case) ...
                            if stream_obj.get_uuid(expected=False) != _request_uuid:
                                stream_obj.set_uuid(None, expected=False)  # clearing UUID
                                stream_obj.set_uuid(_request_uuid, expected=True)  # setting expectations

            # warning: we are not normalizing the hashes, we should do it if this action is called directly
            if yhat_hashes is not None:
                for net_hash in yhat_hashes:
                    if net_hash in self.known_streams:
                        for stream_obj in self.known_streams[net_hash].values():
                            if stream_obj.get_uuid(expected=False) != _request_uuid:
                                stream_obj.set_uuid(None, expected=False)  # clearing UUID
                                stream_obj.set_uuid(_request_uuid, expected=True)  # setting expectations

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

        # searching for the processor-streams of the agent who generated the (inference) data
        processor_streams = self.find_streams(_requester, name_or_group="processor")
        if processor_streams is None or len(processor_streams) == 0:
            self.err("Unexpected confirmation of finished learning")
            return False

        # warning: differently from the case of done_gen, we are not considering the streams generated by the
        # learning agents as something we could use for evaluation (this might be changed in the future)

        # clearing the UUID of the local streams associated to the agent who learned
        for net_hash, stream_dict in processor_streams.items():
            for stream_obj in stream_dict.values():
                stream_obj.set_uuid(None, expected=False)
                stream_obj.set_uuid(None, expected=True)

        # if one or more of my streams where used as arguments of the learning request I did (ask_learn), then their
        # UUID must be cleared...we clear them all
        for net_hash, stream_dict in self.owned_streams.items():
            for stream_obj in stream_dict.values():
                if stream_obj.props.is_public() != self.behaving_in_world():
                    stream_obj.set_uuid(None, expected=False)
                    stream_obj.set_uuid(None, expected=True)
        return True

    def all_asked_finished(self):
        """Check if all the agents who where asked for something (calling "ask_*") are done."""
        return self._agents_who_were_asked == self._agents_who_completed_what_they_were_asked

    def all_engagements_completed(self):
        """Check if all the agents who where asked for engagement ("send_engagement") confirmed ("got_engagement")."""
        return len(self._found_agents) == 0

    def agents_are_waiting(self):
        """Check if at least one agent from self._found_agents is waiting to be considered in order to be added."""
        self.out(f"Current set of {len(self._node_agents_waiting)} connected peer IDs non managed yet: "
                 f"{self._node_agents_waiting}")
        for found_agent in self._found_agents:
            if found_agent in self._node_agents_waiting:
                return True
        return False

    def ask_subscribe(self, agent: str | None = None,
                      stream_hashes: list[str] | None = None, unsubscribe: bool = False):

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        self.deb(f"[ask_subscribe] Involved_agents: {involved_agents}")

        if len(involved_agents) == 0:
            self.deb(f"[ask_subscribe] No involved agents, action ask_gen returns False")
            return False

        # create a copy of the stream hashes, normalizing them in the appropriate way
        stream_hashes_copy: list[str | None] = [None] * len(stream_hashes)
        for i in range(len(stream_hashes_copy)):
            if stream_hashes_copy[i] == "<playlist>":
                # from <playlist> to the current element of the playlist
                stream_hashes_copy[i] = self._preferred_streams[self._cur_preferred_stream]
            else:
                # from a user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                stream_hashes_copy[i] = self.user_stream_hash_to_net_hash(stream_hashes[i])

        # getting properties
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
                     _requester: str | list | None = None, _request_time: float = -1.) -> bool:
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

        # building properties
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

        # adding new streams and subscribing (if compatible with our processor)
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
        """This is a way to get back the confirmation of a completed subscription/unsubscription.

        Args:
            unsubscribe: If this action is to confirm an unsubscription (True) or a subscription (False).
            _requester: The ID of the agent who completed the subscription. Defaults to None.

        Returns:
            True if the subscription/unsubscription confirmation was successfully handled by this agent.
        """
        what = "subscribing" if unsubscribe else "unsubscribing"
        self.out(f"Agent {_requester} finished {what}")

        # remembering that the agent that invoked this action is the one who actually subscribed
        self._agents_who_completed_what_they_were_asked.add(_requester)
        return True

    def record(self, net_hash: str, samples: int = 100, time: float = -1., timeout: float = -1.):
        """Record a stream."""
        assert samples is not None and time is not None and timeout is not None, "Missing basic action information"

        k = self.get_action_step()

        self.out(f"Recording stream {net_hash}")

        if k == 0:

            # getting stream(s)
            _net_hash = self.user_stream_hash_to_net_hash(net_hash)  # in case of ambiguity, it yields the first one
            if _net_hash is None:
                self.err(f"Unknown stream {net_hash}")
                return False
            else:
                net_hash = _net_hash

            stream_src_dict = self.known_streams[net_hash]

            # creating the new recorded stream (same props of the recorded one, just owned now)
            stream_dest_dict = {}
            for name, stream_obj in stream_src_dict.items():
                props = stream_obj.props.clone()
                props.set_group("recorded" + str(self._last_recorded_stream_num))
                stream_dest_dict[name] = BufferedDataStream(props=props, clock=self._node_clock)
            self._last_recorded_stream_dict = stream_dest_dict
            self._last_recording_stream_dict = stream_src_dict

        else:

            # retrieving the stream(s)
            stream_dest_dict = self._last_recorded_stream_dict
            stream_src_dict = self._last_recording_stream_dict

        # recording
        for name, stream_obj in stream_src_dict.items():
            x = stream_obj.get(requested_by="record")
            if x is None:
                self.deb("[record] data sample missing, returning False")
                return False
            else:
                self.deb(f"[record] data_tag: {stream_obj.get_tag()}, data_uuid: {stream_obj.get_uuid()}")
            stream_dest_dict[name].set(x, k)  # saving specific data tags 0, 1, 2, ... #record_steps - 1

        # updating profile
        if self.is_last_action_step():
            self.deb("[record] last action step detected, finishing")

            # dummy get to ensure that the next get will return None (i.e., we only PubSub if somebody restarts this)
            for stream_obj in stream_dest_dict.values():
                stream_obj.get(requested_by="send_stream_samples")

            self.add_streams(list(stream_dest_dict.values()), owned=True)
            self.update_streams_in_profile()
            self.subscribe_to_pubsub_owned_streams()
            self.send_profile_to_all()

            # new recorded stream
            self._last_recorded_stream_num += 1

        return True

    def connect_by_role(self, role: str | list[str], filter_fcn: str | None = None,
                        time: float = -1., timeout: float = -1.):
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

    def find_agents(self, role: str | list[str]):
        """Find an agent whose authority is in the specified range."""

        self.out(f"Finding an available agent whose role is {role}")
        role_list = role if isinstance(role, list) else [role]
        self._found_agents = set()

        for role in role_list:
            role = self.ROLE_STR_TO_BITS[role]
            role_base_int = role & 3

            if role_base_int != 0:
                if role_base_int == self.ROLE_WORLD_AGENT:
                    agents = self.world_agents
                elif role_base_int == self.ROLE_WORLD_MASTER:
                    agents = self.world_masters
                else:
                    return False
            else:
                agents = self.world_agents | self.world_masters

            role = (role >> 2) << 2
            for peer_id, profile in agents.items():
                _role = self.ROLE_STR_TO_BITS[profile.get_dynamic_profile()['connections']['role']]
                _role = (_role >> 2) << 2
                if _role == role:
                    self._found_agents.add(peer_id)  # peer IDs here

        self.deb(f"[find_agents] Found these agents: {self._found_agents}")
        return len(self._found_agents) > 0

    def next_pref_stream(self):
        """Moves to the next stream in the list of preferred ones."""

        if len(self._preferred_streams) == 0:
            self.err(f"Cannot move to the next stream because the list of preferred streams is empty")
            return False

        self._cur_preferred_stream = (self._cur_preferred_stream + 1) % len(self._preferred_streams)
        suffix = ", warning: restarted" if self._cur_preferred_stream == 0 else ""
        self.out(f"Moving to the next preferred stream ({self._preferred_streams[self._cur_preferred_stream]}){suffix}")
        return True

    def first_pref_stream(self):
        """Moves to the next stream in the list of preferred ones."""

        if len(self._preferred_streams) == 0:
            self.err(f"Cannot move to the first stream because the list of preferred streams is empty")
            return False

        self._cur_preferred_stream = 0
        self.out(f"Moving to the first preferred stream ({self._preferred_streams[self._cur_preferred_stream]})")
        return True

    def check_pref_stream(self, what: str = "last") -> bool:
        """Check the current preferred stream."""

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
        """Fill a list with preferred streams."""

        self.out(f"Setting up a list of {len(net_hashes)} preferred streams")
        self._cur_preferred_stream = 0
        self._preferred_streams = []
        self._repeat = repeat
        for i in range(0, self._repeat):
            for net_hash in net_hashes:
                # we are tolerating both peer_id:name_or_group and also peer_id::ps:name_or_group
                components = net_hash.split(":")
                peer_id = components[0]
                name_or_group = components[-1]
                net_hash_to_streams = self.find_streams(peer_id=peer_id, name_or_group=name_or_group)
                for _net_hash in net_hash_to_streams.keys():
                    self._preferred_streams.append(_net_hash)

        return True

    def evaluate(self, stream_hash: str, how: str, steps: int = 100, re_offset: bool = False) -> bool:
        """Compare two signals."""

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
            received_net_hash = self._last_buffered_peer_id_to_info[peer_id]["net_hash"]
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

    def compare_eval(self, cmp: str, thres: float, good_if_true: bool = True) -> bool:
        """After having completed an evaluation."""

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

        # setting recipient in the case of direct messages
        # (differently, in case of pubsub, the agent is already sending messages to all)
        if u_hashes is not None:
            for u_hash in u_hashes:
                if not DataProps.is_pubsub_from_net_hash(u_hash):
                    self._recipients[u_hash] = agent
        if yhat_hashes is not None:
            for yhat_hash in yhat_hashes:
                if not DataProps.is_pubsub_from_net_hash(yhat_hash):
                    self._recipients[yhat_hash] = agent

        # triggering
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
                          ref_uuid: str | None = None) -> bool:
        """Loop on data streams, for learning and/or generation purposes."""

        # getting current step index
        k = self.get_action_step()

        # checking data and creating new buffered streams
        if k == 0:
            self.deb("[__process_streams] First action step")

            # checking data
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

        # generate output
        outputs, data_tag_from_inputs = (
            self.generate(input_net_hashes=u_hashes, first=(k == 0), last=self.is_last_action_step(),
                          ref_uuid=ref_uuid))
        if outputs is None:
            return False
        self.deb(f"[__process_streams] data_tag_from_inputs: {data_tag_from_inputs}")
        if data_tag_from_inputs is None:
            data_tag_from_inputs = -1
            self.deb(f"[__process_streams] data_tag_from_inputs (forced): {data_tag_from_inputs}")

        # learn
        if learn:
            self.deb(f"[__process_streams] learning, step {k}")
            loss_values, data_tags_from_targets = self.learn_generate(outputs=outputs, targets_net_hashes=yhat_hashes)
            self.deb(f"[__process_streams] data_tags_from_targets: {data_tags_from_targets}")

            if loss_values is None:
                return False
            else:
                self.out(f"Losses: {loss_values}")

            # fusing data tags
            data_tags = [data_tag_from_inputs if _data_tag == -1 else _data_tag for _data_tag in data_tags_from_targets]
        else:
            data_tags = [data_tag_from_inputs] * len(outputs)
        self.deb(f"[__process_streams] data_tags (final): {data_tags}")

        # set each data sample in "outputs" to the right stream
        i = 0
        for net_hash, stream_dict in self.proc_streams.items():

            # setting the data sample
            for name, stream_obj in stream_dict.items():

                # public output streams are only considered if the agent IS NOT acting in a world
                # private output streams are only considered if the agent IS acting in a world
                if self.behaving_in_world() != stream_obj.props.is_public():

                    # guessing recipient of the communication
                    if i == 0:
                        self._recipients[net_hash] = recipient \
                            if not DataProps.is_pubsub_from_net_hash(net_hash) else None

                    self.deb(f"[__process_streams] Setting the {i}-th network output to stream with "
                             f"net_hash: {net_hash}, name: {name}")
                    # here we exploit the fact that streams were inserted in order
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
        """Post action to run after at the end of a do_something call, to confirm it."""

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

        # confirming
        if send_back_confirmation:
            if self.set_next_action(peer_id_who_asked, action="done_" + do_what, args={}):
                return True
            else:
                self.err(f"Unable to confirm '{do_what}' to {peer_id_who_asked}")
                return False
        else:
            return True

    def __compare_streams(self, net_hash_a: str, net_hash_b: str,
                          how: str = "mse", steps: int = 100, re_offset: bool = False) -> tuple[float, bool]:
        """Loop on two -buffered- data streams, for comparison purposes, returning a value >= 0."""

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
            # if there is only 1 stream is each group, things are easy
            stream_a = next(iter(stream_dict_a.values()))
            stream_b = next(iter(stream_dict_b.values()))
        elif len(stream_dict_a) == 1 and len(stream_dict_b) > 1:
            # if there is only 1 stream is one of the groups, we look for a compatible stream in the other group,
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
            # if there is only 1 stream is one of the groups, we look for a compatible stream in the other group,
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
            # if both groups have more than a stream, let's give priority to streams with labels to find a match
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
                if _a.dtype == torch.long and _b.dtype == torch.long:  # token IDS
                    return 1. - float((_a == _b).sum().item()) / a.numel()  # accuracy
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
                ret = 1. - float(_a == _b)  # strings (always handled as 'same')
            return ret

        # comparing data (averaging)
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

            # signals or descriptors
            a, a_tag = stream_a[k_a]
            b, b_tag = stream_b[k_b]

            # if the streams do not share the same first tag equal to zero, and we asked to re-offset them,
            # then we force the initial offsets to be zero on both
            # if not, then re-offset the tags
            if k_a == 0 and k_b == 0 and re_offset:
                a_tag_offset = a_tag
                b_tag_offset = b_tag

            # offset-based tags
            a_tag_w_offset = a_tag - a_tag_offset
            b_tag_w_offset = b_tag - b_tag_offset

            # checking
            if a is None:
                self.err("Cannot compare stream samples if the reference stream yields None")
                return -1., False

            # some streams might have been pre-buffered in advance, and have increasing data tags belonging to finite,
            # fixed set (such as 0, 1, 2, ..., N). when continuously streaming them, we will go from tag N to tag 0 at
            # a certain point, which is a "restart".
            # we have to remember that this happened, and we do it for stream "a", our "reference" stream.
            # then, below, we will fix tags on stream "b" if needed, considering that such a restart happened.
            if a_tag_prev is not None and a_tag < a_tag_prev:
                restart_detected = True

            # some streams might have been pre-buffered in advance, and have a fixed data tag (usually -1).
            # being it negative, it will happen that the data tag will be replaced by a clock cycle, but this function
            # does not change clock cycles at all, so all samples will have the exact same data tag.
            # the following code automatically advances the tag by 1 for stream "a", that is expected to be the
            # reference stream (i.e., the one for which the agent has all samples, with no missing data in between)
            if a_tag_prev is not None and a_tag <= a_tag_prev:
                a_tag = a_tag_prev + 1  # fixed tag detected (patching)
                a_tag_w_offset = a_tag - a_tag_offset

            # fixing
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
                        o = o + (1. if how != "mse" else (o / steps) * 1.1)  # don't change k_b, some samples missing
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
                        k_b += 1  # a restart was detected, it means that "stream_b" is behind, let's move it ahead
                elif b_tag_w_offset < a_tag_w_offset:
                    self.deb(f"[__compare_streams] (d) Comparing tags: {a_tag} vs {b_tag} -> too early w.r.t. expected "
                             f"(with offsets: {a_tag_w_offset} vs {b_tag_w_offset}) "
                             f"samples: {a} vs {b}")
                    return -1., False

        self.deb(f"[__compare_streams] Error: {o / steps}")
        # input("*** press enter to continue ***")
        return o / steps, True

    def __involved_agents(self, agent: str | None) -> list[str]:
        peer_id = agent
        involved_agents = [peer_id] if peer_id is not None and peer_id != "<valid_cmp>" else (
            self._valid_cmp_agents) if peer_id is not None and peer_id == "<valid_cmp>" else self._engaged_agents
        if len(involved_agents) == 0:
            self.err("Not engaged to any agents or no agent specified")
        return involved_agents
