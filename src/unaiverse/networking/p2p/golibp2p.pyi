from typing import List
import ctypes

class GoLibP2P:
    def InitializeLibrary(self, max_instances: ctypes.c_int, max_num_channels: ctypes.c_int, max_queue_per_channel: ctypes.c_int, max_msg_size: ctypes.c_int, log_config: bytes) -> None:
        """
        InitializeLibrary(max_instances: ctypes.c_int, max_queue_per_channel: ctypes.c_int, max_num_channels: ctypes.c_int, max_msg_size: ctypes.c_int, enable_logging: ctypes.c_int) -> bytes

        Configures the P2P library.
        """
        ...
    
    def CreateNode(self, instance: ctypes.c_int, node_config_json: bytes) -> ctypes.c_int:
        """
        CreateNode(instance: ctypes.c_int, node_config_json: bytes) -> bytes

        Creates a node in the P2P network and returns a JSON string with node information.
        """
        ...

    def ConnectTo(self, instance: ctypes.c_int, multiaddrs_json: bytes) -> ctypes.c_int:
        """
        ConnectTo(instance: ctypes.c_int, multiaddrs_json: bytes) -> bytes

        Connects to a peer using the provided multiaddress. Returns a JSON string with the result.
        """
        ...

    def StartStaticRelay(self, instance: ctypes.c_int, relay_info_json: bytes) -> ctypes.c_int:
        """
        EnableStaticRelay(instance: ctypes.c_int, relay_info_json: bytes) -> bytes

        Enables (or switches to) a static AutoRelay service for the given relay info.
        Returns a JSON result.
        """
        ...

    def DisconnectFrom(self, instance: ctypes.c_int, peer_id: bytes) -> ctypes.c_int:
        """
        DisconnectFrom(instance: ctypes.c_int, peer_id: bytes) -> bytes

        Disconnects from the given peer id. Returns a JSON result.
        """
        ...

    def GetConnectedPeers(self, instance: ctypes.c_int) -> ctypes.c_int:
        """
        GetConnectedPeers(instance: ctypes.c_int) -> bytes

        Returns a JSON string listing connected peers.
        """
        ...

    def GetRendezvousPeers(self, instance: ctypes.c_int) -> ctypes.c_int:
        """
        GetRendezvousPeers(instance: ctypes.c_int) -> bytes

        Returns a JSON string listing rendezvous peers.
        """
        ...

    def GetNodeAddresses(self, instance: ctypes.c_int, arg: bytes) -> ctypes.c_int:
        """
        GetNodeAddresses(instance: ctypes.c_int, arg: bytes) -> bytes

        Returns the node addresses in a JSON string.
        """
        ...

    def SendMessageToPeer(
        self,
        instance: ctypes.c_int,
        channel: bytes,
        data: ctypes.c_char_p,
        data_len: ctypes.c_int,
    ) -> ctypes.c_int:
        """
        SendMessageToPeer(instance: ctypes.c_int, channel: bytes, data: bytes, data_len: ctypes.c_int) -> bytes
        """
        ...

    def SubscribeToTopic(self, instance: ctypes.c_int, topic_composite_key: bytes) -> ctypes.c_int:
        """
        SubscribeToTopic(instance: ctypes.c_int, topic_composite_key: bytes) -> bytes

        Subscribes to a topic and returns a JSON string with the result.
        """
        ...
    
    def UnsubscribeFromTopic(self, instance: ctypes.c_int, topic_composite_key: bytes) -> ctypes.c_int:
        """
        UnsubscribeFromTopic(instance: ctypes.c_int, topic_composite_key: bytes) -> bytes

        Unsubscribe from a topic and returns a JSON string with the result.
        """
        ...

    def MessageQueueLength(self, instance: ctypes.c_int) -> ctypes.c_int:
        """
        MessageQueueLength(instance: ctypes.c_int) -> ctypes.c_int

        Returns the current length of the message queue.
        """
        ...

    def PopMessages(self, instance: ctypes.c_int) -> ctypes.c_int:
        """
        PopNMessages(instance: ctypes.c_int) -> bytes

        Pops the first message in each channel queue and returns them as a list.
        """
        ...

    def CloseNode(self, instance: ctypes.c_int) -> ctypes.c_int:
        """
        CloseNode(instance: ctypes.c_int) -> bytes

        Closes the node and frees all resources.
        """
        ...

    def FreeString(self, arg: bytes) -> None:
        """
        FreeString(arg: bytes) -> None

        Frees a string previously allocated by the shared library.
        """
        ...

    def FreeInt(self, arg: ctypes.c_int) -> None:
        """
        FreeInt(arg: ctypes.c_int) -> None

        Frees an integer previously allocated by the shared library.
        """
        ...
