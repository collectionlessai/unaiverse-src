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
import io
import os
import json
import torch
import base64
from PIL import Image
from flask_cors import CORS
from threading import Thread
from unaiverse.dataprops import DataProps
import torchvision.transforms as transforms
from unaiverse.streams import BufferedDataStream
from unaiverse.networking.node.node import NodeSynchronizer
from flask import Flask, jsonify, request, send_from_directory


class Server:

    def __init__(self, node_synchronizer: NodeSynchronizer,
                 root: str = '../../../../zoo/debug_viewer/www',
                 port: int = 5001,
                 checkpoints: dict[str, list[dict] | int] | str | None = None,
                 y_range: list[float] | None = None):
        self.node_synchronizer = node_synchronizer
        self.node_synchronizer.using_server = True  # Forcing
        self.root = os.path.join(os.path.dirname(os.path.abspath(__file__)), root)
        self.root_css = self.root + "/static/css"
        self.root_js = self.root + "/static/js"
        self.port = port
        self.app = Flask(__name__, template_folder=self.root)
        CORS(self.app)  # To handle cross-origin requests (needed for development)
        self.register_routes()
        self.thumb_transforms = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64)])
        self.y_range = y_range
        self.visu_name_to_net_hash = {}

        # Loading checkpoints, if needed
        if checkpoints is not None and isinstance(checkpoints, str):  # String: assumed to be a file name
            file_name = checkpoints
            checkpoints = {"checkpoints": None, "matched": -1, "current": 0}
            with open(file_name, 'r') as file:
                checkpoints["checkpoints"] = json.load(file)  # From filename to dictionary
        elif checkpoints is not None:
            checkpoints = {"checkpoints": checkpoints, "matched": -1, "current": 0}
        self.node_synchronizer.server_checkpoints = checkpoints

        # Fixing y_range as needed
        self.y_range = [None, None] if self.y_range is None else self.y_range
        assert len(self.y_range) == 2, "Invalid y_range argument (it must be either None of a list of 2 floats)"

        # Starting a new thread
        thread = Thread(target=self.__run_server)
        thread.start()

    def __run_server(self):
        self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False)  # Run Flask with threading enabled

    def register_routes(self):
        self.app.add_url_rule('/', view_func=self.serve_index, methods=['GET'])
        self.app.add_url_rule('/<path:filename>', view_func=self.serve_root, methods=['GET'])
        self.app.add_url_rule('/static/css/<path:filename>', view_func=self.serve_static_css, methods=['GET'])
        self.app.add_url_rule('/static/js/<path:filename>', view_func=self.serve_static_js, methods=['GET'])
        self.app.add_url_rule('/get_play_pause_status', view_func=self.get_play_pause_status, methods=['GET'])
        self.app.add_url_rule('/ask_to_pause', view_func=self.ask_to_pause, methods=['GET'])
        self.app.add_url_rule('/ask_to_play', view_func=self.ask_to_play, methods=['GET'])
        self.app.add_url_rule('/get_env_name', view_func=self.get_env_name, methods=['GET'])
        self.app.add_url_rule('/get_summary', view_func=self.get_summary, methods=['GET'])
        self.app.add_url_rule('/get_authority', view_func=self.get_authority, methods=['GET'])
        self.app.add_url_rule('/get_behav', view_func=self.get_behav, methods=['GET'])
        self.app.add_url_rule('/get_behav_status', view_func=self.get_behav_status, methods=['GET'])
        self.app.add_url_rule('/get_list_of_agents', view_func=self.get_list_of_agents, methods=['GET'])
        self.app.add_url_rule('/get_list_of_streams', view_func=self.get_list_of_streams, methods=['GET'])
        self.app.add_url_rule('/get_stream', view_func=self.get_stream, methods=['GET'])
        self.app.add_url_rule('/get_console', view_func=self.get_console, methods=['GET'])
        self.app.add_url_rule('/save', view_func=self.save, methods=['GET'])

    @staticmethod
    def pack_data(_data):
        _type = type(_data).__name__ if _data is not None else "none"

        def is_tensor_or_list_of_tensors(_d):
            if isinstance(_d, list) and len(_d) > 0 and isinstance(_d[0], torch.Tensor):
                return True
            elif isinstance(_d, torch.Tensor):
                return True
            else:
                return False

        def is_pil_or_list_of_pils(_d):
            if isinstance(_d, list) and len(_d) > 0 and isinstance(_d[0], Image.Image):
                return True
            elif isinstance(_d, Image.Image):
                return True
            else:
                return False

        # List of pytorch tensors (or nones)
        def encode_tensor_or_list_of_tensors(__data):
            __type = ""

            if isinstance(__data, list) and len(__data) > 0 and isinstance(__data[0], torch.Tensor):
                found_tensor = False
                __data_b64 = []
                for __tensor in __data:
                    if __tensor is not None:
                        if not found_tensor:
                            found_tensor = True
                            __type = "list_" + type(__data[0]).__name__ + "_" + __data[0].dtype.__str__().split('.')[-1]

                        __data_b64.append(base64.b64encode(__tensor.detach().cpu().numpy().tobytes()).decode('utf-8'))
                    else:
                        __data_b64.append(None)  # There might be some None in some list elements...
                if not found_tensor:
                    __type = "none"
                __data = __data_b64

            # Pytorch tensor
            if isinstance(__data, torch.Tensor):
                __type = __data.dtype.__str__().split('.')[-1]
                __data = base64.b64encode(__data.detach().cpu().numpy()).decode('utf-8')

            return __data, __type

        # List of PIL images (or nones)
        def encode_pil_or_list_of_pils(__data):
            __type = ""

            if isinstance(__data, list) and len(__data) > 0 and isinstance(__data[0], Image.Image):
                found_image = False
                _data_b64 = []
                for __img in __data:
                    if __img is not None:
                        if not found_image:
                            found_image = True
                            __type = "list_png"

                        buffer = io.BytesIO()
                        __img.save(buffer, format="PNG", optimize=True, compress_level=9)
                        buffer.seek(0)
                        _data_b64.append(f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}")
                    else:
                        _data_b64.append(None)  # There might be some None in some list elements...
                if not found_image:
                    __type = "none"
                __data = _data_b64

            # Pil image
            if isinstance(__data, Image.Image):
                __type = "png"
                __buffer = io.BytesIO()
                __data.save(__buffer, format="PNG", optimize=True, compress_level=9)
                __data = f"data:image/png;base64,{base64.b64encode(__buffer.read()).decode('utf-8')}"

            return __data, __type

        # In the case of a dictionary, we look for values that are (list of) tensors/images and encode them;
        # we augment the key name adding "-type", where "type" is the type of the packed data
        if _type == "dict":
            keys = list(_data.keys())
            for k in keys:
                v = _data[k]
                if is_tensor_or_list_of_tensors(v):
                    v_encoded, v_type = encode_tensor_or_list_of_tensors(v)
                    del _data[k]
                    k = k + "-" + v_type
                    _data[k] = v_encoded
                elif is_pil_or_list_of_pils(v):
                    v_encoded, v_type = encode_pil_or_list_of_pils(v)
                    del _data[k]
                    k = k + "-" + v_type
                    _data[k] = v_encoded
        else:
            if is_tensor_or_list_of_tensors(_data):
                _data, _data_type = encode_tensor_or_list_of_tensors(_data)
                _type += "_" + _data_type
            elif is_pil_or_list_of_pils(_data):
                _data, _data_type = encode_pil_or_list_of_pils(_data)
                _type += "_" + _data_type
            else:
                pass

        # Generate JSON for the whole data, where some of them might have been base64 encoded (tensors/images)
        return jsonify({"data": _data, "type": _type})

    def serve_index(self):
        return send_from_directory(self.root, 'index.html')

    def serve_root(self, filename):
        return send_from_directory(self.root, filename)

    def serve_static_js(self, filename):
        return send_from_directory(self.root_js, filename)

    def serve_static_css(self, filename):
        return send_from_directory(self.root_css, filename)

    def get_play_pause_status(self):
        ret = {'status': None,
               'still_to_play': self.node_synchronizer.skip_clear_for,
               'time': self.node_synchronizer.clock.get_time(passed=True),
               'y_range': self.y_range,
               'matched_checkpoint_to_show': None,
               'more_checkpoints_available': False}
        if self.node_synchronizer.synch_cycle == self.node_synchronizer.synch_cycles:
            ret['status'] = 'ended'
        elif self.node_synchronizer.step_event.is_set():
            ret['status'] = 'playing'
        elif self.node_synchronizer.wait_event.is_set():
            ret['status'] = 'paused'
        if self.node_synchronizer.server_checkpoints is not None:
            ret['more_checkpoints_available'] = self.node_synchronizer.server_checkpoints["current"] >= 0
            if self.node_synchronizer.server_checkpoints["matched"] >= 0:
                ret['matched_checkpoint_to_show'] = self.node_synchronizer.server_checkpoints["checkpoints"][
                    self.node_synchronizer.server_checkpoints["matched"]]["show"]
        return Server.pack_data(ret)

    def ask_to_play(self):
        steps = int(request.args.get('steps'))
        if steps >= 0:
            self.node_synchronizer.skip_clear_for = steps - 1
        else:
            self.node_synchronizer.skip_clear_for = steps
        self.node_synchronizer.step_event.set()
        return Server.pack_data(self.node_synchronizer.synch_cycle)

    def ask_to_pause(self):
        self.node_synchronizer.skip_clear_for = 0
        return Server.pack_data(self.node_synchronizer.synch_cycle)

    def get_env_name(self):
        return Server.pack_data({"name": self.node_synchronizer.world.get_name(),
                                 "title": self.node_synchronizer.world.get_name()})

    def get_summary(self):
        agent_name = request.args.get('agent_name')
        desc = str(self.node_synchronizer.agent_nodes[agent_name].agent) \
            if agent_name != self.node_synchronizer.world.get_name() else str(self.node_synchronizer.world)
        return Server.pack_data(desc)

    def get_authority(self):
        agent_name = request.args.get('agent_name')
        role = self.node_synchronizer.agent_name_to_profile[agent_name].get_dynamic_profile()['connections']['role']
        authority = 1.0 if "high_authority" in role else 0.0
        return Server.pack_data(authority)

    def get_behav(self):
        agent_name = request.args.get('agent_name')
        if agent_name == self.node_synchronizer.world.get_name():
            behav = self.node_synchronizer.world.behav
        else:
            behav = self.node_synchronizer.agent_nodes[agent_name].agent.behav
        return Server.pack_data(str(behav.to_graphviz().source))

    def get_behav_status(self):
        agent_name = request.args.get('agent_name')
        if agent_name == self.node_synchronizer.world.get_name():
            behav = self.node_synchronizer.world.behav
        else:
            behav = self.node_synchronizer.agent_nodes[agent_name].agent.behav
        state = behav.get_state().id if behav.get_state() is not None else None
        action = behav.get_action().id if behav.get_action() is not None else None
        return Server.pack_data({'state': state, 'action': action,
                                 'state_with_action': behav.get_state().has_action()
                                 if (state is not None) else False})

    def get_list_of_agents(self):
        agents = self.node_synchronizer.agent_nodes
        ret = {"agents": list(agents.keys()), "authorities": [
            1.0 if "teacher" in self.node_synchronizer.agent_name_to_profile[x].
            get_dynamic_profile()['connections']['role'] else 0.0 for x in agents.keys()]}
        return Server.pack_data(ret)

    def get_list_of_streams(self):
        agent_name = request.args.get('agent_name')
        agent = self.node_synchronizer.agent_nodes[agent_name].agent\
            if agent_name != self.node_synchronizer.world.get_name() else (
            self.node_synchronizer.world)
        streams = agent.known_streams
        decoupled_streams = []
        for net_hash, stream_dict in streams.items():
            assert len(stream_dict) <= 2, (f"Agent {agent_name}: "
                                           f"unexpected size of a stream group ({len(stream_dict)}), expected 2. "
                                           f"The net hash is {net_hash} and here is "
                                           f"the corresponding dict: "
                                           f"{str({k: str(v.get_props()) for k, v in stream_dict.items()})}")
            group_name = DataProps.name_or_group_from_net_hash(net_hash)

            found = False
            peer_id = DataProps.peer_id_from_net_hash(net_hash)
            for _agent_name, _agent_node in self.node_synchronizer.agent_nodes.items():
                _agent = _agent_node.agent
                public_peer_id, private_peer_id = _agent.get_peer_ids()
                if peer_id == public_peer_id or peer_id == private_peer_id:
                    group_name = _agent_name.lower() + ":" + group_name
                    self.visu_name_to_net_hash[group_name] = net_hash
                    found = True
                    break
            if not found:
                public_peer_id, private_peer_id = self.node_synchronizer.world.get_peer_ids()
                if peer_id == public_peer_id or peer_id == private_peer_id:
                    group_name = "world" + ":" + group_name
                    self.visu_name_to_net_hash[group_name] = net_hash

            decoupled_streams.append(group_name + " [y]")
            decoupled_streams.append(group_name + " [d]")
        return Server.pack_data(decoupled_streams)

    def get_stream(self):
        agent_name = request.args.get('agent_name')
        stream_name = request.args.get('stream_name')
        since_step = int(request.args.get('since_step'))
        data_id = 0
        if stream_name.endswith(" [y]"):
            group_name = stream_name[0:stream_name.find(" [y]")]
            data_id = 0
        elif stream_name.endswith(" [d]"):
            group_name = stream_name[0:stream_name.find(" [d]")]
            data_id = 1
        else:
            group_name = stream_name

        if agent_name != self.node_synchronizer.world.get_name():
            agent = self.node_synchronizer.agent_nodes[agent_name].agent
            known_streams = self.node_synchronizer.agent_nodes[agent_name].agent.known_streams
        else:
            agent = self.node_synchronizer.world
            known_streams = self.node_synchronizer.world.known_streams

        net_hash = self.visu_name_to_net_hash[group_name]
        stream_objs = list(known_streams[net_hash].values())
        stream_obj = stream_objs[data_id] if data_id < len(stream_objs) else None

        if stream_obj is None:

            # Missing stream
            ks = [agent._node_clock.get_cycle()]
            data = None
            last_k = agent._node_clock.get_cycle()
            props = None
        elif isinstance(stream_obj, BufferedDataStream):

            # Buffered stream
            ks, data, last_k, props = stream_obj.get_since_cycle(since_step)
        else:

            # Not-buffered stream
            sample = stream_obj.get()
            ks = [agent._node_clock.get_cycle()]
            data = [sample] if sample is not None else None
            last_k = agent._node_clock.get_cycle()
            props = stream_obj.get_props()

        # Data is None if the step index (k) of the stream is -1 (beginning), or if stream is disabled
        if data is not None:

            # If data has labeled components (and is not "img" and is not "token_ids"),
            # then we take a decision and convert it to a text string
            if props.is_flat_tensor_with_labels():
                for _i, _data in enumerate(data):
                    data[_i] = props.to_text(_data)

            # If data is of type image, we revert the possibly applied transformation and downscale it
            elif props.is_img():
                for _i, _data in enumerate(data):
                    data[_i] = self.thumb_transforms(_data)

        return Server.pack_data({
            "ks": ks,
            "data": data,
            "last_k": last_k
        })

    def get_console(self):
        agent_name = request.args.get('agent_name')
        last_only = request.args.get('last_only')

        is_world = agent_name == self.node_synchronizer.world.get_name()

        if is_world:
            node = self.node_synchronizer.world_node  # <-- ✅ this must be set in your code
            agent = node.world
            behav = agent.behav
        else:
            node = self.node_synchronizer.agent_nodes[agent_name]
            agent = node.agent
            behav = agent.behav

        state = behav.get_state().id if behav.get_state() is not None else None
        action = behav.get_action().id if behav.get_action() is not None else None

        output_messages = node._output_messages
        output_ids = node._output_messages_ids
        count = node._output_messages_count
        last_pos = node._output_messages_last_pos

        if last_only is None or not last_only:
            return Server.pack_data({
                'output_messages': output_messages,
                'output_messages_count': count,
                'output_messages_last_pos': last_pos,
                'output_messages_ids': output_ids,
                'behav_status': {
                    'state': state,
                    'action': action,
                    'state_with_action': behav.get_state().has_action() if state is not None else False
                }
            })
        else:
            return Server.pack_data({
                'output_messages': [output_messages[last_pos]],
                'output_messages_count': 1,
                'output_messages_last_pos': 0,
                'output_messages_ids': [output_ids[last_pos]],
                'behav_status': {
                    'state': state,
                    'action': action,
                    'state_with_action': behav.get_state().has_action() if state is not None else False
                }
            })

    def save(self):
        return Server.pack_data(self.node_synchronizer.world.env.save())  # TODO
