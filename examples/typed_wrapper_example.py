"""Example demonstrating TypedGoWrapper usage.

This example shows how to use the TypedGoWrapper for cleaner,
type-safe interactions with the Go libp2p library.

Run this after installing the unaiverse package.
"""

from unaiverse.networking.p2p import P2P, P2PConfig, GoLibConfig, TypedGoWrapper
import tempfile
import os

def main():
    """Demonstrate TypedGoWrapper usage."""

    # Create temporary directory for node identity
    with tempfile.TemporaryDirectory() as temp_dir:
        identity_dir = os.path.join(temp_dir, "node_keys")

        # Configure P2P
        p2p_config = P2PConfig(
            max_instances=1,
            max_channels=10,
            max_queue_per_channel=10,
            max_message_size=1024 * 1024,
            enable_logging=True
        )

        go_config = GoLibConfig(
            identity_dir=identity_dir,
            predefined_port=0,  # Random port
            relay_enable_client=False,
            relay_enable_service=False,
            network_isolated=False,
            dht_enabled=False,
        )

        # Create P2P node
        print("=" * 80)
        print("Creating P2P node...")
        print("=" * 80)

        node = P2P(p2p_config=p2p_config, go_p2p_config=go_config)

        # Access the typed wrapper
        wrapper: TypedGoWrapper = P2P._wrapper

        print("\n" + "=" * 80)
        print("Example 1: Get Node Addresses (Typed)")
        print("=" * 80)

        # Get node addresses using wrapper
        addresses_response = wrapper.get_node_addresses(instance=0, peer_id="")

        if addresses_response.is_success:
            addresses = addresses_response.unwrap()
            print(f"✅ Node addresses: {addresses}")
        else:
            print(f"❌ Error: {addresses_response.message}")

        print("\n" + "=" * 80)
        print("Example 2: Get Connected Peers (Typed)")
        print("=" * 80)

        # Get connected peers
        peers_response = wrapper.get_connected_peers(instance=0)

        if peers_response.is_success:
            peers = peers_response.unwrap()
            print(f"✅ Connected peers: {len(peers)}")
            for peer in peers:
                print(f"  - {peer.id} ({peer.direction})")
        else:
            print(f"❌ Error: {peers_response.message}")

        print("\n" + "=" * 80)
        print("Example 3: Pop Messages (Typed, with Empty Check)")
        print("=" * 80)

        # Pop messages
        messages_response = wrapper.pop_messages(instance=0)

        if messages_response.is_empty:
            print("✅ No messages in queue (this is normal)")
        elif messages_response.is_success:
            messages = messages_response.unwrap()
            print(f"✅ Received {len(messages)} messages")
            for msg in messages:
                print(f"  From: {msg.from_peer}")
                print(f"  Data length: {len(msg.data)} bytes")
        else:
            print(f"❌ Error: {messages_response.message}")

        print("\n" + "=" * 80)
        print("Example 4: Message Queue Length")
        print("=" * 80)

        # Get queue length (returns int directly, not a response)
        queue_length = wrapper.message_queue_length(instance=0)
        print(f"✅ Message queue length: {queue_length}")

        print("\n" + "=" * 80)
        print("Example 5: Subscribe to Topic (Typed)")
        print("=" * 80)

        # Subscribe to a topic
        topic = f"{node.peer_id}::ps:test_topic"
        subscribe_response = wrapper.subscribe_to_topic(instance=0, topic=topic)

        if subscribe_response.is_success:
            print(f"✅ Subscribed: {subscribe_response.unwrap()}")
        else:
            print(f"❌ Error: {subscribe_response.message}")

        print("\n" + "=" * 80)
        print("Example 6: Error Handling with unwrap()")
        print("=" * 80)

        # Try to connect to invalid peer (will fail)
        try:
            connect_response = wrapper.connect_to(
                instance=0,
                multiaddrs=["/ip4/127.0.0.1/tcp/9999/p2p/InvalidPeerID"]
            )
            # This will raise ValueError if connection fails
            peer_info = connect_response.unwrap()
            print(f"✅ Connected to: {peer_info.id}")
        except ValueError as e:
            print(f"❌ Expected error (invalid peer): {e}")

        print("\n" + "=" * 80)
        print("Comparison: Old vs New API")
        print("=" * 80)

        print("\nOLD (two-step pattern):")
        print("  result_ptr = P2P.libp2p.GetNodeAddresses(...)")
        print("  result = P2P._type_interface.from_go_ptr_to_json(result_ptr)")
        print("  if result.get('state') == 'Error': ...")
        print("  addrs = result.get('message')")

        print("\nNEW (single typed call):")
        print("  response = wrapper.get_node_addresses(...)")
        print("  addrs = response.unwrap()  # Type: List[str]")

        print("\n✅ Benefits:")
        print("  - 50% less code")
        print("  - Full type hints")
        print("  - IDE autocomplete")
        print("  - Automatic error handling")

        # Cleanup
        print("\n" + "=" * 80)
        print("Cleaning up...")
        print("=" * 80)

        close_response = wrapper.close_node(instance=0)
        if close_response.is_success:
            print(f"✅ {close_response.unwrap()}")
        else:
            print(f"❌ Error closing node: {close_response.message}")

if __name__ == "__main__":
    main()
