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
import torch
import functools
from typing import Callable
from PIL.Image import Image
from unaiverse.hsm import Action
from unaiverse.stats import Stats
from unaiverse.clock import clock
from unaiverse.utils.logger import log
from unaiverse.interaction import Interaction
from unaiverse.agent_basics import AgentBasics
from unaiverse.networking.p2p.messages import Msg
from unaiverse.custom import Custom, GenException
from unaiverse.streams.dataprops import DataProps
from unaiverse.streams.streams import BufferedStream, Stream


def action(func: Callable) -> Callable:
    """Decorator (@action) that standardizes action method arguments.

    Applied to action methods to:
    - Resolve agent references from name to peer_id.
    - Mark the function as an action (sets ``_is_action = True``).
    - Handle unexpected exceptions.

    Args:
        func: The action method to decorate.

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        self: Agent
        resolved_kwargs = {}

        if func.__name__ not in Custom.SPECIAL_ACTION_NAMES:  # Special actions (like 'send') resolve by themselves
            for key, value in kwargs.items():

                if key in Custom.AGENT_ARG_NAMES:
                    if isinstance(value, str):
                        resolved = self.resolve_agent_ref(value)
                        resolved_kwargs[key] = resolved if resolved is not None else value
                    elif isinstance(value, list | tuple):
                        _values = list(value)  # Shallow
                        for i, _value in enumerate(_values):
                            _resolved = self.resolve_agent_ref(_value)
                            if _resolved is not None:
                                _values[i] = _resolved
                        resolved_kwargs[key] = _values
                    else:
                        resolved_kwargs[key] = value

                elif key in Custom.STREAM_ARG_NAMES:
                    public = not self.behaving_in_world()
                    if isinstance(value, str):
                        resolved = self.resolve_stream_ref(value, public)
                        resolved_kwargs[key] = resolved if resolved is not None else value
                    elif isinstance(value, (list, tuple)):
                        _values = list(value)  # Shallow
                        for i, _value in enumerate(_values):
                            _resolved = self.resolve_stream_ref(_value, public)
                            if _resolved is not None:
                                _values[i] = _resolved
                        resolved_kwargs[key] = _values
                    else:
                        resolved_kwargs[key] = value

                else:
                    resolved_kwargs[key] = value
        else:
            resolved_kwargs = kwargs

        # Calling the action
        try:
            ret = await func(self, *args, **resolved_kwargs)
            if ret is None:
                log.critical(f"Invalid action '{func.__name__}' detected, it must return True/False "
                             "(it is tolerated to return non-None stuff and interpreted as True, "
                             "but not if it returns None, as happens here)")
        except Exception as e:
            if not isinstance(e, AssertionError):
                log.error(f"Action '{func.__name__}' failed since it raised an exception: {e}")  # No-Debug
                import traceback  # Debug
                log.error(f"{traceback.format_exc()}")  # Debug
            else:
                log.error(f"Action '{func.__name__}' failed since it raised an AssertionError")  # No-Debug
                import traceback
                log.error(f"{traceback.format_exc()}")
            ret = False
        return ret

    wrapper._is_action = True
    return wrapper


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
        self._last_recorded_count = 0

        # Status variables (assumed to start with "_"): Playlist
        self._preferred_streams = []  # List of preferred streams
        self._cur_preferred_stream = 0  # ID of the current preferred stream from the list
        self._repeat = 1  # Number of repetitions of the playlist

        # Stats
        self.stats = Stats(is_world=False)
        self.overwrite_stats = False  # Whether to overwrite stats when receiving the next STATS_RESPONSE from the world

    def collect_and_store_own_stats(self) -> None:
        """Collects this agent's own stats and pushes them to the stats recorder."""
        if self.stats is None:
            return

        _, own_private_pid = self.get_peer_ids()
        t = clock.get_time_ms()
        try:
            info = self.node_conn['p2p_world'].get_connected_peers_info()
            peers_list = [i['id'] for i in info]
            self.stats.store_stat('connected_peers', peers_list, group_key=own_private_pid, timestamp=t)
        except Exception as e:
            self.stats.store_stat('connected_peers', [], group_key=own_private_pid, timestamp=t)
            log.error(f"[Stats] Error collecting and storing own stats, clearing: {e}")

        try:
            behav = self.behav
            assert behav is not None
            self.stats.store_stat('state', behav.get_state_name(), group_key=own_private_pid, timestamp=t)
            self.stats.store_stat('action', behav.get_action_name(), group_key=own_private_pid, timestamp=t)
            self.stats.store_stat('last_action', behav.get_last_completed_action_name(),
                                  group_key=own_private_pid, timestamp=t)
        except Exception as e:
            log.error(f"[Stats] Error storing HSM stats: {e}")

    async def send_stats_to_world(self) -> None:
        """Sends the agent's currently buffered stats to the world and clears them (async)."""
        if not self.in_world():
            log.debug("[send_stats_to_world] Not in a world, skipping stats send.")
            return

        world_peer_id = self.node_conn.get_world_peer_id()
        if world_peer_id is None:
            log.error("Agent in world, but world_peer_id is None.")
            return

        self.collect_and_store_own_stats()  # update own stats
        assert self.stats is not None
        payload = self.stats.get_payload_for_world()
        if not payload:
            log.debug("[send_stats_to_world] No stats to send.")
            return

        # Send all stats
        log.misc(f"Sending stats update to world {world_peer_id}...")
        if not (await self.node_conn.send(world_peer_id,
                                          channel_trail=None,
                                          content=payload,
                                          content_type=Msg.STATS_UPDATE)):
            log.error("Failed to send stats update to world.")

    def update_stats_view(self, received_view: dict, overwrite: bool = False) -> None:
        """Updates the stats view with the data received from the world.

        Args:
            received_view: The view dictionary received from the world.
            overwrite: Whether to overwrite existing entries. Defaults to False.
        """
        assert self.stats is not None
        self.stats.update_view(received_view, overwrite)

    async def suggest_role_to_world(self, agent: str | None, role: str) -> bool:
        """Suggests a role change for one or more agents to the world master. It iterates through the involved agents,
        checks if their current role differs from the suggested one, and sends a role suggestion message to the
        world master (async).

        Args:
            agent: The ID of the agent or a wildcard to suggest the role for.
            role: The new role to suggest (as a string).

        Returns:
            True if the suggestion was sent successfully, False otherwise.
        """
        log.misc("Suggesting role to world")

        agents = self.__involved_agents(agent)
        role_bits = (self.ROLE_STR_TO_BITS[role] >> 2) << 2

        content = []

        for _agent in agents:
            cur_role_bits = self.ROLE_STR_TO_BITS[self.all_agents[_agent].get_dynamic_profile()['connections']['role']]
            cur_role_bits = (cur_role_bits >> 2) << 2
            if cur_role_bits == role_bits:
                log.misc(f"Not suggesting to change the role of {_agent} "
                         f"since it has already such a role")
            else:
                log.misc(f"Suggesting to change the role of {_agent} to {self.ROLE_BITS_TO_STR[role_bits]}")
                content.append({'peer_id': _agent, 'role': role_bits})

        if len(content) > 0:
            world_peer_id = self.node_conn.get_world_peer_id()
            if not (await self.node_conn.send(world_peer_id, channel_trail=None,
                                              content=content,
                                              content_type=Msg.ROLE_SUGGESTION)):
                log.error("Failed to send role suggestion to the world")
                return False
        return True

    async def suggest_badges_to_world(self, agent: str | None = None,
                                      score: float = -1.0, badge_type: str = "completed",
                                      badge_description: str | None = None) -> bool:
        """Suggests one or more badges to the world master for specific agents. This is typically used to reward agents
        for completing tasks, such as for a competition. It sends a message with the badge details, including the score
        and type, to the world master (async).

        Args:
            agent: The ID of the agent or a wildcard for which to suggest the badge.
            score: The score associated with the badge.
            badge_type: The type of badge (e.g., 'completed').
            badge_description: An optional description for the badge.

        Returns:
            True if the badge suggestion was sent successfully, False otherwise.
        """
        log.misc("Suggesting one or more badges to world")

        if score < 0.:
            log.error("Invalid score (did you specify the 'score' argument? it must be positive)")
            return False

        agents = self.__involved_agents(agent)
        world_peer_id = self.node_conn.get_world_peer_id()

        if badge_type not in Agent.BADGE_TYPES:
            log.error(f"Unknown badge type: {badge_type}")
            return False

        list_of_badge_dictionaries = []
        for peer_id in agents:
            list_of_badge_dictionaries.append({'peer_id': peer_id,
                                               'score': score,
                                               'badge_type': badge_type,
                                               'badge_description': badge_description,
                                               'agent_token': self.node_conn.get_last_token(peer_id)})

        if not (await self.node_conn.send(world_peer_id, channel_trail=None,
                                          content=list_of_badge_dictionaries,
                                          content_type=Msg.BADGE_SUGGESTIONS)):
            log.error("Failed to send badge suggestions to the world")
            return False
        else:
            return True

    def remove_peer_from_agent_status_attrs(self, peer_id: str) -> None:
        """Removes a peer from all agent-level status tracking sets (overrides parent) (async).

        Args:
            peer_id: The peer ID to remove from status tracking structures.
        """
        super().remove_peer_from_agent_status_attrs(peer_id)
        self._available = len(self._engaged_agents) == 0

    def reset_agent_status_attrs(self) -> None:
        """Resets all agent-level status attributes to their initial values (overrides parent)."""
        super().reset_agent_status_attrs()  # this sets status vars to [], {}, 0, 0., False, in function of their type
        self._available = True
        self._repeat = 1
        self._last_recorded_stream_num = 1
        self._last_recorded_count = 0

    @action
    async def send(self, action_name: str | None = None,
                   target: str | list[str] | None = None,
                   action_kwargs: dict | None = None,
                   streams: dict[str, str | list[str]] | list[str] | None = None,
                   data_samples: list | dict[str, torch.Tensor | Image | str] | None = None,
                   num_steps: int = -1,
                   timeout: float = -1.,
                   from_state: str | None = None,
                   to_state: str | None = None,
                   callback: str | None = None,
                   forced_uuid: str | None = "do_not_force",
                   id: str | None = "random",
                   copy_sys: bool = False,
                   wait_completion: bool = False,
                   volatile: bool = False,
                   interaction: Interaction | None = None) -> bool:
        """Send an interaction request to one or more target agents (async).

        Accepts either a pre-built :class:`Interaction` object *or* raw arguments from which one will be created.

        Args:
            action_name: The action name (str) if building one from raw args.
                Ignored when a pre-built Interaction is provided.
            target: Peer ID (or wildcard such as ``"<valid_cmp>"``) of the target
                agent. It can be a list of agents. Ignored when a pre-built Interaction is provided.
            action_kwargs: Action arguments dict.  Ignored when a pre-built Interaction
                is provided.
            streams: List of ``(stream_hash, num_samples)`` tuples (or just stream_hash if num_samples = 1).
                Ignored when a pre-built Interaction is provided. A lot of freedom here: stream_hash can be a net hash
                or a user hash. It is also fine to specify just the stream name or the stream
                group, then the user hash or the net hash will be automatically estimated.
            data_samples: List of actual data samples. Ignored when a pre-built Interaction is provided. It can also be
                a dictionary mapping one or more stream user hashes to data samples that will be artificially loaded on
                such streams, so that the 'streams' argument above is actually automatically populated.
                It can also be a string representing the payload that encodes a list of samples, if it was received
                from the net.
            num_steps: Number of steps the interaction spans (used when building from raw args).
                Ignored when a pre-built Interaction is provided.
            timeout: Maximum time for the interaction. Ignored when a pre-built Interaction is provided.
            from_state: Optional source state. Ignored when a pre-built Interaction provided.
            to_state: Optional destination state. Ignored when a pre-built Interaction provided.
            callback: Callback action (will be called when the interaction completes).
            forced_uuid: UUID of the interaction to explicitly force, use carefully.
            id: Optional ID of the interaction. Ignored when a pre-built Interaction is provided.
            copy_sys: True if the interaction must copy the last added stream data with system UUID
                to prepare initial data form the current interaction.
            wait_completion: If True, this action returns True if and only if we get a feed of interaction completion
                from all the involved agents (Default: False).
            volatile: If True, it is marked so that the recipient is asked to not
                send back any status about its completion.
            interaction: The interaction triggered by the system to run this action (automatically set).

        Returns:
            True if at least of the target agents was reached.
        """

        # This action can only be triggered by the system or by the developer, no ways
        if interaction is not None and not interaction.is_system():
            return False

        if wait_completion and volatile:
            raise GenException("You cannot wait for completion a volatile interaction (it will not return any updates)")

        if wait_completion and interaction is None:
            raise GenException("You cannot wait for completion by directly calling this action, without an interaction"
                               " (the state of the action needs to be saved in the interaction object)")

        if interaction is None:  # This happens when calling 'send' from the code, e.g., inside another action
            system_interaction = None
            first_run = True
        else:
            system_interaction = interaction
            first_run = system_interaction.get_mark() is None

        if first_run:
            target = self.__involved_agents(target)
            sent_interaction = await self._send(None, action_name, target, action_kwargs, streams,
                                                data_samples, num_steps, timeout, from_state, to_state, callback,
                                                forced_uuid, id, copy_sys, volatile)
            if wait_completion:
                assert system_interaction is not None
                assert isinstance(system_interaction.action_ref, Action)
                system_interaction.action_ref.set_default_timeout()  # This will make the action pedantic
                system_interaction.set_mark(sent_interaction)  # First run, Saving the interaction that was sent
                return False
            else:
                if system_interaction:
                    assert isinstance(system_interaction.action_ref, Action)
                    system_interaction.action_ref.set_timeout(-1.)  # Clear timeouts to ensure it is not pedantic
                return sent_interaction is not None
        else:
            assert system_interaction is not None
            sent_interaction = system_interaction.get_mark()  # Loading the interaction that was sent at the first run
            sent_interaction: Interaction
            if sent_interaction.is_completed():
                return True  # The mark will be automatically cleared by the Action class
            else:
                return False

    @action
    async def all_sent_completed(self, action_name: str | None = None) -> bool:
        """True iff every sent interaction with the given `action_name` (or any, if None) has been
        completed by the IM. Negation of "are there any still-pending sends out there?". Useful as an
        HSM exit predicate after `send(action_name="...", ...)` was dispatched without `wait_completion`."""
        for inter in self.im.sent.values():
            if action_name is None or inter.action_name == action_name:
                if not inter.is_completed():
                    return False
        return True

    @action
    async def process(self, interaction: Interaction | None = None) -> bool:
        """Runs one inference step: reads from stdin, calls the processor, and writes to stdout (async).

        Returns:
            True if the step ran successfully, False otherwise.
        """
        # Getting UUID of the interaction (used for tags and stdout)
        if interaction is not None:
            uuid = interaction.uuid
        else:
            uuid = Custom.SYSTEM_INTERACTION_UUID

        # Possibly re-binding stdin to cope with human processor (it might also let the action fail, if needed)
        if not self.__rebind_stdin_if_human(interaction):
            return False

        # Getting input data from the input stream
        input_data = self.stdin.get(uuid=uuid, requested_by="process")  # Use kwargs

        if input_data is None:
            return False

        # Getting data tag
        data_tag = self.stdin.get_tag(uuid=uuid)

        if data_tag is None:
            return False

        # Post get actions to finalize what possibly started with the previous call to the rebind operation
        # (do this only after having received valid data from stdin)
        self.__rebind_stdin_if_human_finalizer(interaction)

        # Customizable input hook
        try:
            input_data = self.hook_proc_tweak_inputs(input_data)
            if input_data is None:
                return False
        except Exception as e:
            log.error(f"Error tweaking the processor inputs: {e}")
            return False

        # Processing data
        try:
            output_data = self.proc(*input_data, first=self.get_action_step() == 0, last=self.is_last_action_step())

            if not isinstance(output_data, tuple):
                output_data = (output_data,)
        except Exception as e:
            log.error(f"Error running the processor: {e}")
            return False

        # Customizable output hook
        try:
            output_data = self.hook_proc_tweak_outputs(output_data)
        except Exception as e:
            log.error(f"Error tweaking the processor outputs: {e}")
            return False

        # Pushing output data to the output stream
        self.stdout.set(output_data, data_tag=data_tag, uuid=uuid)  # Use kwargs for data_tag and uuid
        return True

    @action
    async def learn(self, interaction: Interaction | None = None) -> bool:
        """Runs one learning step: first performs inference, then runs a backward pass (async).

        Returns:
            True if the learning step ran successfully, False otherwise.
        """
        if (self.proc_opts is None or self.proc_opts['optimizer'] is None
                or self.proc_opts['losses'] is None
                or len(self.proc_opts['losses']) == 0):
            log.misc(f"Current processor has no learning skills (or learning options not specified)")
            return False

        assert interaction is not None

        # Inference first
        if await self.process(interaction):

            # Getting the UUID of the interaction
            uuid = interaction.uuid

            # Getting input data from the input stream
            target_data = self.stdtar.get(uuid=uuid, requested_by="learn")  # Use kwargs

            # Learning from the last performed inference and the current targets
            try:
                assert self.proc is not None
                loss_values = self.proc.learn_backward(target_data)
            except Exception as e:
                log.error(f"Error while learning: {e}")
                return False

            # Printing
            log.user(f"Losses: {[x.item() for x in loss_values]}, "
                     f"Step: {self.get_action_step()}, Tag: {self.stdin.get_tag()}, "
                     f"Last Step: {interaction.num_steps-1}", rep=True),
            return True
        else:
            return False

    @action
    async def show(self, interaction: Interaction | None = None):
        """A generic action that exploit the information in the interaction object for printing purposes, commonly used
        (by overriding it) to handle interaction-completion callbacks (async).

        Args:
            interaction: The interaction object that triggered this action.

        Returns:
            True all the times.
        """
        if interaction is None:
            return False
        log.debug(f"[show] The following interaction was completed: "
                  f"{interaction.to_code_str(True, True)}")
        return True

    @action
    async def set_engaged_partner(self, agent: str | list[str] | set[str] | None, clear_found: bool = True) -> bool:
        """Virtually forces the engagement with a single agent (or a group of agents), clearing all existing
        engagements and results of previous find operations (async).

        Args:
            agent: The agent or the list of agents to engage (if None, we only clear the list of engaged partners).
            clear_found: Clears results of previously run find operations (True by default).

        Returns:
            True all the times.
        """
        if clear_found:
            self._found_agents.clear()
        self._engaged_agents.clear()
        if agent is not None:
            if isinstance(agent, str):
                agent = [agent]
            for a in agent:
                self._engaged_agents.add(a)
        self._available = len(self._engaged_agents) == 0
        return True

    @action
    async def send_engage(self) -> bool:
        """Offer engagement to the agents whose identifiers are in self._found_agents (async).

        Returns:
            True if engagement requests were successfully sent to at least one found agent, False otherwise.
        """
        if len(self._found_agents) > 0:
            log.misc(f"Sending engagement request to {', '.join([x for x in self._found_agents])}")
        my_role_str = self.node_profile.get_dynamic_profile()['connections']['role']
        ret = await self.send(target=self._found_agents, action_name="engage",
                              action_kwargs={"sender_role": my_role_str}, wait_completion=True,
                              interaction=self.im.get_current())
        if ret:
            engaged = self.last_sent_interaction.target if self.last_sent_interaction is not None else None
            if (engaged is not None) and isinstance(engaged, list) and len(engaged) > 0:
                for eng in engaged:
                    self._engaged_agents.add(eng)

                    # Marking this agent as not available since it engaged with another one
                    self._available = False

                    # Removing the agent from the list of asked agents
                    self._found_agents.discard(eng)
                return True
            else:
                return False
        else:
            return False

    @action
    async def engage(self, acceptable_role: str | None = None, sender_role: str | None = None,
                     interaction: Interaction | None = None) -> bool:
        """Receive engagement from another agent whose authority is in the specified range (async).

        Args:
            acceptable_role: The role that the sender must have for engagement to be accepted. Defaults to None.
            sender_role: The role of the agent sending the engagement request. Defaults to None.
            interaction: The interaction triggered by the agent requesting engagement (automatically set).

        Returns:
            True if the engagement was successfully received and confirmed, False otherwise.
        """
        _requester = interaction.requester if interaction is not None else None
        log.misc(
            f"Getting engagement from {_requester}, whose role is {sender_role} (looking for {acceptable_role})")
        if _requester not in self.world_agents and _requester not in self.world_masters:
            log.error(f"Unknown agent: {_requester}")
            return False

        if sender_role is None:
            log.error(f"Unknown role of {_requester}")
            return False

        if acceptable_role is None:
            log.error(f"Invalid acceptable role: {acceptable_role}")
            return False

        # Confirming
        if self._available:
            acceptable_role_int = self.ROLE_STR_TO_BITS[acceptable_role]
            if "~" not in acceptable_role:
                sender_role_int = (self.ROLE_STR_TO_BITS[sender_role] >> 2) << 2
            else:
                sender_role_int = self.ROLE_STR_TO_BITS[sender_role]

            if acceptable_role_int == sender_role_int:
                self._engaged_agents.add(_requester)

                # Marking this agent as not available since it engaged with another one
                self._available = False
                return True
            else:
                log.error(f"Cannot engage to {_requester}")
                return False
        else:
            log.error(f"Cannot engage to {_requester}")
            return False

    @action
    async def send_disengage(self, send_disconnection_too: bool = False) -> bool:
        """Ask for disengagement (async).

        Args:
            send_disconnection_too: Whether to send a disconnect-suggestion together with the disengagement.

        Returns:
            True if disengagement requests were successfully sent to at least one engaged agent, False otherwise.
        """
        if len(self._engaged_agents) > 0:
            log.misc(f"Sending disengagement request to {', '.join([x for x in self._engaged_agents])}")
        if await self._send(target=list(self._engaged_agents), action_name="disengage",
                            action_kwargs={"disconnect_too": send_disconnection_too}):
            self._engaged_agents.clear()  # There is no "got_disengagement"
            return True
        else:
            log.error(f"Unable to send disengagement to {self._engaged_agents}")
            return False

    @action
    async def disengage(self, disconnect_too: bool = False, interaction: Interaction | None = None) -> bool:
        """Get a disengagement request from an agent (async).

        Args:
            disconnect_too: Whether to disconnect the agent who sent the disengagement.
            interaction: The interaction triggered by the agent requesting disengagement (automatically set).

        Returns:
            True if the disengagement request was successfully processed, False otherwise.
        """

        _requester = interaction.requester if interaction is not None else None
        log.misc(f"Getting a disengagement request from {_requester}")
        if _requester not in self.world_agents and _requester not in self.world_masters:
            log.error(f"Unknown agent: {_requester}")
            return False

        if _requester not in self._engaged_agents:
            log.error(f"Not previously engaged to {_requester}")
            return False

        self._engaged_agents.discard(_requester)  # Remove if present

        if disconnect_too:
            if interaction is not None:
                interaction.send_status = False  # This will avoid sending back a confirmation for this interaction
            await self.node_purge_fcn(_requester)

        # Marking this agent as available if not engaged to any agent
        self._available = len(self._engaged_agents) == 0
        return True

    @action
    async def disengage_all(self) -> bool:
        """Disengage all the previously engaged agents (async).

        Returns:
            True if the disengagement procedure was successfully executed, False otherwise.
        """
        log.misc(f"Disengaging all agents")
        self._engaged_agents = set()

        # Marking this agent as available
        self._available = True
        return True

    @action
    async def disconnect(self, agent: str) -> bool:
        """Disconnects an agent (async).

        Args:
            agent: The peer ID of the agent to disconnect.

        Returns:
            Always True.
        """
        log.misc(f"Disconnecting agent: {agent}")
        await self.node_purge_fcn(agent)  # This will also call remove_agent, that will call remove_streams
        return True

    @action
    async def disconnect_by_role(self, role: str | list[str], disengage_too: bool = False) -> bool:
        """Disconnects from all agents that match a specified role (async).
        It finds the agents and calls the node's purge function on each.

        Args:
            role: A string or list of strings representing the role(s) of agents to disconnect from.
            disengage_too: Also forces the sending of a disengagement (before disconnecting - default False).

        Returns:
            Always True.
        """
        log.misc(f"Disconnecting agents with role: {role}")
        if disengage_too:
            await self.send_disengage(send_disconnection_too=True)
        if await self.find_agents(role):
            found_agents = copy.deepcopy(self._found_agents)
            for agent in found_agents:
                await self.node_purge_fcn(agent)  # This will also call remove_agent, that will call remove_streams
        return True

    @action
    @action
    async def disconnected(self, agent: str | None = None, handshake_completed: bool = False) -> bool:
        """Checks if a specific set of agents (by ID or wildcard) are no longer connected to the agent.
        It returns False if any of the specified agents are still connected (async).

        Args:
            agent: The ID of the agent or a wildcard to check.
            handshake_completed: If True, only consider agents that have completed the handshake.

        Returns:
            True if all involved agents are disconnected, False otherwise.

        """

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        log.misc(f"Checking if all these agents are not connected to me anymore: {involved_agents}")
        all_disconnected = True
        for agent in involved_agents:
            if handshake_completed:
                if agent in self.all_agents:
                    all_disconnected = False
                    break
            else:
                if agent in self.all_agents or self.node_conn.is_connected(agent):
                    all_disconnected = False
                    break
        return all_disconnected

    @action
    async def received_some_asked_data(self, processing_fcn: str | None = None, data_type: str | None = None) -> bool:
        """Checks if any of the agents that were previously asked for data (e.g., via `ask_gen`) have sent a stream
        sample back. Optionally, it can process the received data with a specified function (async).

        Args:
            processing_fcn: The name of a function to process the received data.
            data_type: Optional. A string with the type of data to consider (discarding the rest).

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
                log.error(f"Processing function not found: {processing_fcn}")

        got_something = False
        sent = list(self.im.sent.values()) + list(self.im.sent_recently_completed)
        for interaction in sent:
            uuid = interaction.uuid
            agents = interaction.target
            for agent in agents:
                net_hash_to_stream_dict = self.find_streams(agent, "processor", discard_owned=True)
                for stream_dict in net_hash_to_stream_dict.values():
                    for stream_obj in stream_dict.values():
                        assert stream_obj.props is not None
                        if data_type is not None and stream_obj.props.data_type != data_type:
                            continue
                        assert stream_obj.props is not None
                        if not stream_obj.props.is_public():
                            data = stream_obj.get("received_some_asked_data", uuid=uuid)
                            data_tag = stream_obj.get_tag(uuid=uuid)

                            if data is not None:
                                if _processing_fcn is None:
                                    return True
                                else:
                                    got_something = True
                                    _processing_fcn: Callable
                                    _processing_fcn(agent, stream_obj.props, data, data_tag)
        return got_something

    @action
    async def nop(self, message: str | None = None) -> bool:
        """Do nothing (async).

        Args:
            message: An optional message to print. Defaults to None.

        Returns:
            Always True.
        """
        if message is not None:
            log.misc(message)
        return True

    @action
    async def connected(self, agent: str | list[str] | None = None, handshake_completed: bool = False) -> bool:
        """Checks if an agent is connected to us or not.

        Args:
            agent: The agent to check (or None if there are other already set engaged agents).
            handshake_completed: If True, only consider agents that have completed the handshake.

        Returns:
            True if all the requested agents are indeed connected.
        """

        # - if "agent" is a peer ID, the involved agents will be a list with one element.
        # - if "agent" is a known wildcard, as "<valid_cmp>", then involved agents will be self._valid_cmp_agents
        # - if "agent" is None, then the current agent in self._engaged_agents will be returned
        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        log.misc(f"Checking if all these agents are connected to me now: {involved_agents}")

        for agent in involved_agents:
            if handshake_completed:
                if agent not in self.all_agents:
                    return False
            else:
                if not self.node_conn.is_connected(agent):
                    return False
        return True

    @action
    async def all_asked_finished(self) -> bool:
        """Checks if all agents that were previously asked to perform a task (e.g., generate or learn) have sent a
        completion confirmation. It compares the set of agents asked with the set of agents that have completed
        the task (async).

        Returns:
            True if all agents are done, False otherwise.
        """
        return self._agents_who_were_asked == self._agents_who_completed_what_they_were_asked

    @action
    async def all_engagements_completed(self) -> bool:
        """Checks if all engagement requests that were sent have been confirmed. It returns True if there are no agents
        remaining in the `_found_agents` list, implying all have been engaged with or discarded (async).

        Returns:
            True if all engagements are complete, False otherwise.

        """
        return len(self._found_agents) == 0

    @action
    async def agents_are_waiting(self) -> bool:
        """Checks if there are any agents who have connected but have not yet been fully processed or added to the
        agent's known lists. This indicates that new agents are waiting to be managed (async).

        Returns:
            True if there are waiting agents, False otherwise.
        """
        log.misc(f"Current set of {len(self.node_agents_waiting)} connected peer IDs non managed yet: "
                 f"{list(self.node_agents_waiting.keys())}")
        for found_agent in self._found_agents:
            if found_agent in self.node_agents_waiting:
                return True
        return False

    @action
    async def send_subscribe(self, agent: str | None = None,
                             stream_hashes: list[str] | None = None, unsubscribe: bool = False) -> bool:
        """Requests a remote agent or a group of agents to subscribe to or unsubscribe from a list of specified PubSub
        streams. It normalizes the stream hashes and sends an action request containing the stream properties (async).

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
        log.debug(f"[send_subscribe] Involved_agents: {involved_agents}")

        if len(involved_agents) == 0:
            log.debug(f"[send_subscribe] No involved agents, action ask_gen returns False")
            return False

        # Create a copy of the stream hashes, normalizing them in the appropriate way
        stream_hashes_copy = self.__normalize_user_hash(stream_hashes)

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
        log.misc(f"Asking {', '.join(involved_agents)} to {what} {stream_hashes}")
        self._agents_who_completed_what_they_were_asked = set()
        self._agents_who_were_asked = set()
        correctly_asked = []
        for agent in involved_agents:
            if await self._send(target=agent, action_name="subscribe",
                                action_kwargs={"stream_owners": stream_owners,
                                               "stream_props": stream_props,
                                               "unsubscribe": unsubscribe}):
                self._agents_who_were_asked.add(agent)
                ret = True
            else:
                what = "subscribe" if not unsubscribe else "unsubscribe"
                log.error(f"Unable to ask {agent} to {what}")
                ret = False
            log.debug(f"[send_subscribe] Asking {agent} returned {ret}")
            if ret:
                correctly_asked.append(agent)

        log.debug(f"[send_subscribe] Overall, the action send_subscribe (unsubscribe: {unsubscribe})"
                  f" will return {len(correctly_asked) > 0}")
        return len(correctly_asked) > 0

    # noinspection PyUnusedLocal
    @action
    async def subscribe(self, stream_owners: list[str] | None = None, stream_props: list[str] | None = None,
                        unsubscribe: bool = False, interaction: Interaction | None = None) -> bool:
        """Executes a subscription or unsubscription request received from another agent. It processes the stream
        properties, adds or removes the streams from the agent's known streams, and handles the underlying PubSub topic
        subscriptions (async).

        Args:
            stream_owners: A list of peer IDs who own the streams.
            stream_props: A list of JSON-serialized stream properties.
            unsubscribe: A boolean to indicate unsubscription.
            interaction: The interaction that triggered this subscription.

        Returns:
            True if the action is successful, False otherwise.
        """
        if stream_props is None or stream_owners is None:
            log.debug("[subscribe] Called without specifying required arguments")
            return False

        log.debug(f"[subscribe] unsubscribe: {unsubscribe}, "
                  f"stream_owners: {stream_owners}, stream_props: ... ({len(stream_props)} props)")

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
                log.error(f"Expecting a pubsub stream, got a stream named {props.get_name()} "
                          f"(group is {props.get_group()}), which is not pubsub")
                return False

        # Adding new streams and subscribing (if compatible with our processor)
        for stream_owner, prop_dict, prop_obj in zip(stream_owners, props_dicts, props_objs):
            if not unsubscribe:
                if not (await self.add_compatible_streams(peer_id=stream_owner, streams_in_profile=[prop_dict],
                                                          buffered=False, public=False)):
                    log.error(f"Unable to add a pubsub stream ({prop_obj.get_name()}) from agent {stream_owner}: "
                              f"no compatible streams were found")
            else:
                if not (await self.remove_streams(peer_id=stream_owner, name=prop_obj.get_name())):
                    log.misc(f"Unable to unsubscribe from pubsub stream ({prop_obj.get_name()}) "
                             f"of agent {stream_owner}")
        return True

    @action
    async def record(self, interaction: Interaction | None = None,
                     record_uuid: str | None = "see_interaction_uuid") -> bool:
        """Records `interaction.num_steps` samples from the interaction's stdin stream group into a
        new owned `BufferedStream` group and exposes it via pubsub. Multistep by virtue of the
        interaction's `num_steps`; the framework's readiness gate already pauses execution on
        no-fresh-data cycles.

        Args:
            interaction: The driving interaction (auto-injected by the dispatch layer).
            record_uuid: Which `uuid` to read each source stream under. The default sentinel
                `"see_interaction_uuid"` means "use `interaction.uuid`" — the right choice when the
                source agent is a peer that knows about this interaction and publishes under it
                (process-style flow). Pass `record_uuid=None` (or any other explicit uuid string)
                when the source publishes under a different uuid — for instance when recording a
                world stream that has no per-interaction publish, in which case data lives in the
                source's `data_by_uuid[None]` slot.

        Triggered from HSM transits via:

            behav.add_transit("init", "next_state", action="record",
                              args={"streams": {"stdin": ["<world>:group"]},
                                    "num_steps": "<wildcard_or_int>",
                                    "record_uuid": None})
        """
        if interaction is None:
            return False

        read_uuid = interaction.uuid if record_uuid == "see_interaction_uuid" else record_uuid
        k = self.get_action_step()

        if k == 0:
            dest = {}
            for s_obj in interaction.stream_proxy.values():
                if not isinstance(s_obj, Stream):
                    continue
                assert s_obj.props is not None
                props = s_obj.props.clone()
                props.set_group("recorded" + str(self._last_recorded_stream_num))
                dest[s_obj.props.get_name()] = BufferedStream(props=props)
            self._last_recorded_stream_dict = dest

        dest = self._last_recorded_stream_dict

        for s_obj in interaction.stream_proxy.values():
            if not isinstance(s_obj, Stream):
                continue
            x = s_obj.get(uuid=read_uuid, requested_by="record")
            if x is None:
                return False
            assert s_obj.props is not None
            dest_stream = dest[s_obj.props.get_name()]
            if k % 10 == 0:
                log.user(f"Recording Stream: {dest_stream.props.get_group()}.{dest_stream.props.get_name()}, "
                         f"Step: {k}, Last: {interaction.num_steps - 1}")
            dest_stream.set(x, k)

        if self.is_last_action_step():
            for s_obj in dest.values():
                log.user(f"Recording Stream: {s_obj.props.get_name()}, Step: {k}, Last: {interaction.num_steps - 1}")
                s_obj.is_read_only = True
                s_obj.get(requested_by="send_stream_samples")
            self.add_streams(list(dest.values()), owned=True)
            self.update_streams_in_profile()
            await self.subscribe_to_pubsub_owned_streams()
            await self.send_profile_to_all()
            self._last_recorded_stream_num += 1
            self._last_recorded_stream_dict = None
        return True

    @action
    async def connect_to(self, peer_id: str):
        addresses = self.node_conn.get_addrs(peer_id)
        if not self.node_conn.is_connected(peer_id):
            log.misc(f"Asking to get in touch with {peer_id}...")
            peer_id = await self.node_ask_to_get_in_touch_fcn(addresses=addresses, public=False)
            if peer_id is not None:
                return True
            else:
                return False
        else:
            log.misc(f"Not-asking to get in touch with {peer_id}, "
                     f"since I am already connected to the corresponding peer...")
            return True

    @action
    async def connect_by_role(self, role: str | list[str], filter_fcn: str | None = None) -> bool:
        """Finds and attempts to connect with agents whose profiles match a specific role. It can be optionally
        filtered by a custom function. It returns True if at least one valid agent is found (async).

        Args:
            role: The role or list of roles to search for.
            filter_fcn: The name of an optional filter function.

        Returns:
            True if at least one agent is found and a NEW connection request is made, False otherwise.
        """
        log.misc(f"Asking to get in touch with all agents whose role is {role}")

        self._found_agents.clear()
        role_list = role if isinstance(role, list) else [role]
        at_least_one_is_valid = False

        for role in role_list:
            role = self.ROLE_STR_TO_BITS[role]

            found_addresses1, found_peer_ids1 = self.node_conn.find_addrs_by_role(Agent.ROLE_WORLD_MASTER | role,
                                                                                  return_peer_ids_too=True,
                                                                                  discard_yourself=True)
            found_addresses2, found_peer_ids2 = self.node_conn.find_addrs_by_role(Agent.ROLE_WORLD_AGENT | role,
                                                                                  return_peer_ids_too=True,
                                                                                  discard_yourself=True)
            found_addresses = found_addresses1 + found_addresses2
            found_peer_ids = found_peer_ids1 + found_peer_ids2

            if filter_fcn is not None:
                if hasattr(self, filter_fcn):
                    filter_fcn = getattr(self, filter_fcn)
                    if callable(filter_fcn):
                        found_addresses, found_peer_ids = filter_fcn(found_addresses, found_peer_ids)
                else:
                    log.error(f"Filter function not found: {filter_fcn}")

            log.misc(f"Found addresses ({len(found_addresses)}) with role: {role}")
            for f_addr, f_peer_id in zip(found_addresses, found_peer_ids):
                if not self.node_conn.is_connected(f_peer_id):
                    log.misc(f"Asking to get in touch with {f_peer_id}...")
                    peer_id = await self.node_ask_to_get_in_touch_fcn(addresses=f_addr, public=False)
                    if peer_id is not None:
                        at_least_one_is_valid = True
                        self._found_agents.add(peer_id)
                else:
                    log.misc(f"Not-asking to get in touch with {f_addr}, "
                             f"since I am already connected to the corresponding peer...")
                    peer_id = f_peer_id
                log.misc(f"...returned {peer_id}")
        return at_least_one_is_valid

    @action
    async def find_agents(self, role: str | list[str], engage: bool = False, handshake_completed: bool = False) -> bool:
        """Locally searches through the agent's known peers (world and public agents) to find agents with a specific
        role. It populates the `_found_agents` set with the peer IDs of matching agents (async).

        Args:
            role: The role or list of roles to search for.
            handshake_completed: If True, only consider agents that have completed the handshake.
            engage: If you want to force the found agents to be the ones that you are engaged with.

        Returns:
            True if at least one agent is found, False otherwise.
        """
        log.misc(f"Finding an available agent whose role is {role}")
        self._found_agents.clear()
        self._found_agents.update(self.get_agents_by_role(role, handshake_completed=handshake_completed))

        log.debug(f"[find_agents] Found these agents: {self._found_agents}")
        if engage:
            self._engaged_agents = copy.deepcopy(self._found_agents)
        return len(self._found_agents) > 0

    @action
    async def next_pref_stream(self) -> bool:
        """Moves the internal pointer to the next stream in the list of preferred streams, which is often used for
        playlist-like operations. It wraps around to the beginning if it reaches the end (async).

        Returns:
            True if the move is successful, False if the list is empty.
        """
        if len(self._preferred_streams) == 0:
            log.error(f"Cannot move to the next stream because the list of preferred streams is empty")
            return False

        self._cur_preferred_stream = (self._cur_preferred_stream + 1) % len(self._preferred_streams)
        suffix = ", warning: restarted" if self._cur_preferred_stream == 0 else ""
        log.misc(
            f"Moving to the next preferred stream ({self._preferred_streams[self._cur_preferred_stream]}){suffix}")
        return True

    @action
    async def first_pref_stream(self) -> bool:
        """Resets the internal pointer to the first stream in the list of preferred streams. This is useful for
        restarting a playback or processing loop (async).

        Returns:
            True if the move is successful, False if the list is empty.
        """
        if len(self._preferred_streams) == 0:
            log.error(f"Cannot move to the first stream because the list of preferred streams is empty")
            return False

        self._cur_preferred_stream = 0
        log.misc(f"Moving to the first preferred stream ({self._preferred_streams[self._cur_preferred_stream]})")
        return True

    @action
    async def check_pref_stream(self, what: str = "last") -> bool:
        """Checks the position of the current preferred stream within the list. It can check if it's the first, last,
        or if it has completed a full round, among other checks (async).

        Args:
            what: A string specifying the type of check to perform (e.g., 'first', 'last', 'last_round').

        Returns:
            True if the condition is met, False otherwise.
        """
        valid = ['first', 'last', 'not_first', 'not_last', 'last_round', 'not_last_round', 'last_song', 'not_last_song']
        assert what in valid, f"The what argument can only be one of {valid}"

        log.misc(f"Checking if the current preferred playlist item "
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
        else:
            return False

    @action
    async def set_pref_streams(self, net_hashes: list[str], repeat: int = 1) -> bool:
        """Fills the agent's list of preferred streams (a playlist). It can repeat the playlist a specified number of
        times and resolves user-provided stream hashes to their full network hashes (async).

        Args:
            net_hashes: A list of stream hashes to add to the playlist.
            repeat: The number of times to repeat the playlist.

        Returns:
            Always True.
        """
        log.misc(f"Setting up a list of {len(net_hashes)} preferred streams")
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

    @action
    async def evaluate(self, stream_hash: str, how: str, steps: int = 100, re_offset: bool = False,
                       agents_to_evaluate: str | list[str] | None = None) -> bool:
        """Evaluates the performance of agents that have completed a generation task. It compares the generated data
        from each agent with a local stream (which can be a ground truth or reference stream) using a specified
        comparison method (async).

        Args:
            stream_hash: The hash of the local stream to use for comparison.
            how: The name of the comparison method to use.
            steps: The number of steps to perform the evaluation.
            re_offset: A boolean to indicate whether to re-offset the streams.
            agents_to_evaluate: peer ID or list of peer IDs of agents that returned streams that were buffered, and
                that we are going to evaluate (default is None, meaning the agents in
                self._agents_who_completed_what_they_were_asked are considered).

        Returns:
            True if the evaluation is successful, False otherwise.
        """
        if not self.buffer_generated_by_others:
            log.error("Cannot evaluate if not buffering data generated by others")
            return False

        if stream_hash == "<playlist>":
            net_hash = self._preferred_streams[self._cur_preferred_stream]
        else:
            net_hash = self.user_stream_hash_to_net_hash(stream_hash)

        assert net_hash is not None

        # Agents to evaluate
        if agents_to_evaluate is None:
            agents_to_evaluate = self._engaged_agents
        else:
            if isinstance(agents_to_evaluate, str):
                agents_to_evaluate = [agents_to_evaluate]

        self._eval_results = {}
        log.debug(f"[evaluate] Agents to evaluate: {agents_to_evaluate}")
        for peer_id in agents_to_evaluate:
            if peer_id not in self.last_buffered_peer_id_to_info:
                log.error(f"Missing buffered stream for {peer_id}, cannot evaluate it (skipping)")
                continue
            received_net_hash = self.last_buffered_peer_id_to_info[peer_id]["net_hash"]
            log.misc(f"Comparing {net_hash} with {received_net_hash}")
            eval_result, ret = self.__compare_streams(net_hash_a=net_hash,
                                                      net_hash_b=received_net_hash,
                                                      how=how, steps=steps, re_offset=re_offset)
            log.misc(f"Result of the comparison: {eval_result}")
            if not ret:
                return False
            else:
                peer_id = DataProps.peer_id_from_net_hash(received_net_hash)
                self._eval_results[peer_id] = eval_result

        return True

    @action
    async def compare_eval(self, cmp: str, thres: float, good_if_true: bool = True) -> bool:
        """Compares the results of a previous evaluation to a given threshold or finds the best result among all
        agents. It can check for minimum, maximum, or simple threshold-based comparisons, and it populates a list of
        'valid' agents that passed the comparison (async).

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

        if len(self._eval_results) == 0:
            log.user("No results to evaluate!")

        for agent_peer_id, eval_result in self._eval_results.items():
            if cmp not in ["min", "max"]:
                log.misc(f"Checking if result {eval_result} {cmp} {thres}, for agent {agent_peer_id}")
            else:
                if thres >= 0:
                    log.misc(f"Checking if result {eval_result} is the {min_or_max} so far, "
                             f"only if {leq_or_geq} {thres}, for agent {agent_peer_id}")
                else:
                    log.misc(
                        f"Checking if result {eval_result} is the {min_or_max} so far, for agent {agent_peer_id}")

            if eval_result < 0.:
                log.user(f"Invalid evaluation result: {eval_result}")
                return False

            owner_account = self.all_agents[agent_peer_id].get_static_profile()['email']
            agent_name = self.all_agents[agent_peer_id].get_static_profile()['node_name']

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
                        msgs.append(f"Agent {owner_account}/{agent_name} passed with {alias} {eval_result}/{thres}")
                        self._valid_cmp_agents.add(agent_peer_id)
                    else:
                        msgs.append(f"Agent {owner_account}/{agent_name} did not pass, {alias} {eval_result}/{thres}")
                else:
                    if outcome:
                        msgs.append(f"Agent {owner_account}/{agent_name} did not pass, {alias} {eval_result}/{thres}")
                    else:
                        msgs.append(f"Agent {owner_account}/{agent_name} passed with {alias} {eval_result}/{thres}")
                        self._valid_cmp_agents.add(agent_peer_id)
            else:
                if ((cmp == "min" and (thres < 0 or eval_result <= thres) and
                     (eval_result < best_so_far or best_so_far < 0)) or
                        (cmp == "max" and (thres < 0 or eval_result >= thres) and
                         (eval_result > best_so_far or best_so_far < 0))):
                    best_so_far = eval_result
                    self._valid_cmp_agents = {agent_peer_id}
                    msgs = [f"The best agent is {owner_account}/{agent_name}"]
                else:
                    msgs = [f"No best agent found for the considered threshold ({thres})"]

        for msg in msgs:
            log.user(msg)

        if len(self._valid_cmp_agents) == 0:

            # # cheating (hack):
            # self._valid_cmp_agents.append(agent_peer_id)
            # log.misc(", ".join(msgs))
            # return True
            return False
        else:
            return True

    def resolve_stream_ref(self, ref: str, public: bool) -> str | None:
        """Resolve a stream name to a stream user or net hash (heuristic, gives priority to owned streams).

        Args:
            ref: A stream name, without peer IDs, just the name.
            public: A boolean to inform the resolve procedure whether it must look for streams among public agents
                (True) or world-related agents (False).

        Returns:
            The stream user hash or net hash if found, None otherwise.
        """

        if ref == "<playlist>":
            return self._preferred_streams[self._cur_preferred_stream]
        else:
            return super().resolve_stream_ref(ref, public)

    def __compare_streams(self, net_hash_a: str, net_hash_b: str,
                          how: str = "mse", steps: int = 100, re_offset: bool = False) -> tuple[float, bool]:
        """A private helper method that compares two buffered data streams based on a specified metric (e.g., MSE,
        max accuracy). It handles stream compatibility checks, data retrieval, and the actual comparison, returning a
        dissimilarity score.

        Args:
            net_hash_a: The network hash of the first stream.
            net_hash_b: The network hash of the second stream.
            how: The comparison metric ('mse', 'max', 'geq-X').
            steps: The number of samples to compare.
            re_offset: A boolean to re-align stream tags before comparison.

        Returns:
            A tuple containing the dissimilarity score and a success flag (e.g., `(0.5, True)`).
        """
        if net_hash_a not in self.known_streams:
            log.error(f"Unknown stream (net_hash_a): {net_hash_a}")
            return -1., False

        if net_hash_b not in self.known_streams:
            log.error(f"Unknown stream (net_hash_b): {net_hash_b}")
            return -1., False

        if steps <= 0:
            log.error(f"Invalid number of steps: {steps}")
            return -1., False

        if how not in ["mse", "max"] and not how.startswith("geq"):
            log.error(f"Data can be compared by MSE, or by comparing the argmax ('max'), or comparing the number "
                      f"of corresponding bits (obtained by 'geq-X', where '-X' is a number). Unknown: {how})")
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
            log.error(f"Cannot find the data stream to consider in the comparison, {net_hash_a}")
            return -1., False
        if stream_b is None:
            log.error(f"Cannot find the data stream to consider in the comparison, {net_hash_b}")
            return -1., False

        if not isinstance(stream_a, BufferedStream):
            log.error(f"Can only compare buffered streams and {net_hash_a} is not buffered")
            return -1., False

        if not isinstance(stream_b, BufferedStream):
            log.error(f"Can only compare buffered streams and {net_hash_b} is not buffered")
            return -1., False

        if steps > len(stream_a) and steps > len(stream_b):
            log.error(f"Cannot compare streams for {steps} steps, since both of them are shorter "
                      f"(length of the first stream is {len(stream_a)}, of the second stream is {len(stream_b)})")
            return -1., False

        if not stream_a.get_props().is_compatible(stream_b.get_props()):
            log.error(f"Cannot compare incompatible streams")
            return -1., False

        def compare(_a: torch.Tensor | str, _b: torch.Tensor | str, _how: str = "mse") -> float:
            """Compare two samples of signals or descriptors, returning a dissimilarity score >= 0."""

            assert _how in ['mse', 'max', 'same'] or _how.startswith("geq"), f"Invalid comparison in terms of {_how}"

            if isinstance(_a, torch.Tensor) and isinstance(_b, torch.Tensor):
                if _a.shape != _b.shape:
                    return 1.  # Mismatching
                if _a.dtype == torch.long and _b.dtype == torch.long:  # Token IDS
                    return 1. - float((_a == _b).sum().item()) / _a.numel()  # Accuracy
                elif _how == "mse":
                    ret = torch.nn.functional.mse_loss(_a, _b, reduction='mean').item()
                elif _how == "max":
                    ret = 1. - float((torch.argmax(_a) == torch.argmax(_b)).sum().item())
                elif _how == "same":
                    ret = 1. - float(torch.eq(_a, _b).sum()) / _a.numel()
                else:
                    thres = float(_how[3:])
                    ret = 1. - float(torch.sum((_a > thres) == (_b > thres)).item()) / _a.numel()
            else:
                ret = 1. - float(_a == _b)  # Strings (always handled as 'same')
            return ret

        data_uuid_a = None
        data_uuid_b = None
        uuids_a = list(stream_a.get_data_uuids())
        uuids_b = list(stream_b.get_data_uuids())
        if len(uuids_a) > 0 and len(uuids_b) > 0:
            intersection = [uuid for uuid in uuids_a if uuid in uuids_b]
            if len(intersection) > 0:
                data_uuid_a = intersection[-1]
                data_uuid_b = data_uuid_a
            else:
                data_uuid_a = uuids_a[-1]
                data_uuid_b = uuids_b[-1]
        elif len(uuids_a) > 0:
            data_uuid_a = uuids_a[-1]
        elif len(uuids_b) > 0:
            data_uuid_b = uuids_b[-1]

        stream_a.restart(uuid=data_uuid_a)
        stream_b.restart(uuid=data_uuid_b)

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
            a, a_tag = stream_a[(k_a, data_uuid_a)]
            b, b_tag = stream_b[(k_b, data_uuid_b)]

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
                log.error("Cannot compare stream samples if the reference stream yields None")
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
                a_tag_prev: int
                a_tag = int(a_tag_prev) + 1  # Fixed tag detected (patching)
                a_tag_w_offset = a_tag - a_tag_offset

            # Fixing
            if b is None:
                o += 1. if how != "mse" else (o / steps) * 1.1
                log.user(f"Comparing: the second stream yields None")
            else:
                if b_tag_w_offset == a_tag_w_offset:
                    o += compare(a, b, how)
                    k_b += 1
                    log.user(f"Comparing tags by {how}: {a_tag} vs {b_tag}, samples: {a} vs {b}")
                elif b_tag_w_offset > a_tag_w_offset:
                    if not restart_detected:
                        o += 1. if how != "mse" else (o / steps) * 1.1  # Don't change k_b, some samples missing
                        log.user(f"Comparing tags by {how}: {a_tag} vs {b_tag} -> "
                                 f"expected one was missing, samples: {a} vs {b}")
                    else:
                        o += 1. if how != "mse" else (o / steps) * 1.1
                        log.user(f"Comparing tags by {how}: {a_tag} vs {b_tag} -> "
                                 f"expected one was missing, samples: {a} vs {b}")
                        k_b += 1  # A restart was detected, it means that "stream_b" is behind, let's move it ahead
                elif b_tag_w_offset < a_tag_w_offset:
                    log.user(
                        f"Comparing tags by {how}: {a_tag} vs {b_tag} -> too early w.r.t. expected, "
                        f"samples: {a} vs {b}")
                    return -1., False

        log.user(f"Comparing error: {o / steps}")

        # Input("*** press enter to continue ***")
        return o / steps, True

    def __involved_agents(self, agent: str | None | list[str]) -> list[str]:
        """A private helper method that resolves an agent ID or a wildcard into a list of specific peer IDs.
        It can resolve a single agent, a group of agents that passed a previous comparison (`<valid_cmp>`), or all
        currently engaged agents.

        Args:
            agent: The agent ID or wildcard string.

        Returns:
            A list of peer IDs corresponding to the involved agents.
        """
        if isinstance(agent, list):
            return agent
        if isinstance(agent, set):
            return list(agent)
        peer_id = agent
        engaged_or_found = self._engaged_agents if len(self._engaged_agents) > 0 else self._found_agents
        involved_agents = [peer_id] if peer_id is not None and peer_id != "<valid_cmp>" else (
            self._valid_cmp_agents) if peer_id is not None and peer_id == "<valid_cmp>" else engaged_or_found
        if len(involved_agents) == 0:
            log.error("Not engaged to any agents, no previously searched agent, or no agent specified")
        return list(involved_agents)

    def __normalize_user_hash(self, net_hashes: list[str] | None) -> list[str]:
        """Normalizes user-provided stream hashes to their full network-hash equivalents.

        Handles the ``<playlist>`` wildcard and converts ``peer_id:name_or_group`` shorthands to full
        ``peer_id::ps:name_or_group`` net hashes.

        Args:
            net_hashes: A list of user-provided stream hashes to normalize, or None.

        Returns:
            A list of resolved network hashes (empty list if input is None).
        """
        if net_hashes is None:
            return []
        net_hashes_copy = []
        for i in range(len(net_hashes)):
            if net_hashes[i] == "<playlist>":

                # From <playlist> to the current element of the playlist
                net_hashes_copy.append(self._preferred_streams[self._cur_preferred_stream])
            else:

                # From user specified hash to a net hash (e.g., peer_id:name_or_group to peer_id::ps:name_or_group)
                net_hashes_copy.append(self.user_stream_hash_to_net_hash(net_hashes[i]))
        return net_hashes_copy

    def __rebind_stdin_if_human(self, interaction):

        # If not human or no interaction provided, return the original UUID
        if not self.is_human() or interaction is None:
            return True

        # First of clear, clearing residuals of interactions that were completed in the past
        to_remove = []

        in_world = self.behaving_in_world()
        world_peer_id = self.get_connection_pool_manager().get_world_peer_id()

        for _peer_id, _interaction in self.proc_human_peer_id_to_interaction.items():

            # If an interaction is completed, we clear if from the internal map.
            # If we get a system interaction while in a world, then we clear the previously stored requests (since
            # we have likely stored them while in other states).
            # Warning: there could be an issue if the same "process" supports both system and user interactions
            # (both solid and dashed)
            if _interaction.is_completed() or (_peer_id == world_peer_id and in_world and interaction.is_system()):
                to_remove.append(_peer_id)
        for _peer_id in to_remove:
            del self.proc_human_peer_id_to_interaction[_peer_id]

        # Then, we detect who is the current partner of the interaction, if and only if there exist a "partner"-like
        # wildcard. This is important to detect system interactions triggering 'process', that will then be used to
        # send a 'process' request to the partner.
        wildcards = self.behav.get_wildcards() if in_world else self.behav_lone_wolf.get_wildcards()
        system_interaction_will_be_for = None
        if Custom.PARTNER_WILDCARD in wildcards:
            system_interaction_will_be_for = wildcards[Custom.PARTNER_WILDCARD]

        # If a dashed interaction exists, the system interaction must be blocked
        # Whenever we get a system interaction, if we already have stored non-system ones, we kill the action
        if interaction.is_system():
            if (system_interaction_will_be_for is not None and
                    system_interaction_will_be_for in self.proc_human_peer_id_to_interaction):
                return False

        # Whenever we get a non-system interaction for a human, we store it in its own dictionary
        if not interaction.is_system():

            # If we are in a world, the reference peer ID is the one of the world, no matter who sent the interaction
            peer_id = interaction.requester if not in_world else self.get_connection_pool_manager().get_world_peer_id()
            self.proc_human_peer_id_to_interaction[peer_id] = interaction

        # Forcing default stdin binding (discarding the interaction-provided ones)
        self.set_default_stdin_binding()
        return True

    def __rebind_stdin_if_human_finalizer(self, interaction: Interaction | None):

        # If not human or no interaction provided, stop here
        if not self.is_human() or interaction is None:
            return

        if interaction.requester in self.proc_human_peer_id_to_interaction:
            del self.proc_human_peer_id_to_interaction[interaction.requester]

        # Clearing data from the stdin, to avoid garbage residuals
        streams_by_user_hash = self.get_default_stdin_bindings()
        for stream in streams_by_user_hash.values():
            stream.clear_data(interaction.uuid)
