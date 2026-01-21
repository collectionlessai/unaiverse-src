from typing import List


class GoLibP2P:
    def InitializeLibrary(self, max_instances: int, max_num_channels: int, max_queue_per_channel: int, max_msg_size: int, log_config: bytes) -> None:
        """
        InitializeLibrary(max_instances: int, max_queue_per_channel: int, max_num_channels: int, max_msg_size: int, enable_logging: int) -> bytes

        Configures the P2P library.
        """
        ...
    
    def CreateNode(self, instance: int, node_config_json: bytes) -> int:
        """
        CreateNode(instance: int, node_config_json: bytes) -> bytes

        Creates a node in the P2P network and returns a JSON string with node information.
        """
        ...

    def ConnectTo(self, instance: int, multiaddrs_json: bytes) -> int:
        """
        ConnectTo(instance: int, multiaddrs_json: bytes) -> bytes

        Connects to a peer using the provided multiaddress. Returns a JSON string with the result.
        """
        ...

    def StartStaticRelay(self, instance: int, relay_info_json: bytes) -> int:
        """
        EnableStaticRelay(instance: int, relay_info_json: bytes) -> bytes

        Enables (or switches to) a static AutoRelay service for the given relay info.
        Returns a JSON result.
        """
        ...

    def DisconnectFrom(self, instance: int, peer_id: bytes) -> int:
        """
        DisconnectFrom(instance: int, peer_id: bytes) -> bytes

        Disconnects from the given peer id. Returns a JSON result.
        """
        ...

    def GetConnectedPeers(self, instance: int) -> int:
        """
        GetConnectedPeers(instance: int) -> bytes

        Returns a JSON string listing connected peers.
        """
        ...

    def GetRendezvousPeers(self, instance: int) -> int:
        """
        GetRendezvousPeers(instance: int) -> bytes

        Returns a JSON string listing rendezvous peers.
        """
        ...

    def GetNodeAddresses(self, instance: int, arg: bytes) -> int:
        """
        GetNodeAddresses(instance: int, arg: bytes) -> bytes

        Returns the node addresses in a JSON string.
        """
        ...

    def SendMessageToPeer(
        self,
        instance: int,
        channel: bytes,
        data: bytes,
        data_len: int,
    ) -> int:
        """
        SendMessageToPeer(instance: int, channel: bytes, data: bytes, data_len: int) -> bytes
        """
        ...

    def SubscribeToTopic(self, instance: int, topic_composite_key: bytes) -> int:
        """
        SubscribeToTopic(instance: int, topic_composite_key: bytes) -> bytes

        Subscribes to a topic and returns a JSON string with the result.
        """
        ...
    
    def UnsubscribeFromTopic(self, instance: int, topic_composite_key: bytes) -> int:
        """
        UnsubscribeFromTopic(instance: int, topic_composite_key: bytes) -> bytes

        Unsubscribe from a topic and returns a JSON string with the result.
        """
        ...

    def MessageQueueLength(self, instance: int) -> int:
        """
        MessageQueueLength(instance: int) -> int

        Returns the current length of the message queue.
        """
        ...

    def PopMessages(self, instance: int) -> int:
        """
        PopNMessages(instance: int) -> bytes

        Pops the first message in each channel queue and returns them as a list.
        """
        ...

    def CloseNode(self, instance: int) -> int:
        """
        CloseNode(instance: int) -> bytes

        Closes the node and frees all resources.
        """
        ...

    def FreeString(self, arg: bytes) -> None:
        """
        FreeString(arg: bytes) -> None

        Frees a string previously allocated by the shared library.
        """
        ...

    def FreeInt(self, arg: int) -> None:
        """
        FreeInt(arg: int) -> None

        Frees an integer previously allocated by the shared library.
        """
        ...
