# Refactoring Example: Using TypedGoWrapper

This shows how to refactor existing P2P methods to use the new `TypedGoWrapper`.

## Example: `start_lib` method (Node Creation)

### Before (Current Implementation)

```python
@model_validator(mode="after")
def start_lib(self) -> Self:
    """Initializes and starts a new libp2p node."""

    # ... validation code ...

    logger.info(f"🐍 Creating Node (Instance ID: {self._instance})...")
    try:
        # TWO-STEP PATTERN - manual conversions
        result_ptr = P2P.libp2p.CreateNode(
            P2P._type_interface.to_go_int(self._instance),
            P2P._type_interface.to_go_json(self.go_p2p_config.model_dump_json()),
        )
        result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

        # MANUAL NULL CHECK
        if result is None:
            err_msg = "Received null result from Go CreateNode."
            logger.error(f"[Instance {self._instance}] {err_msg}")
            raise P2PError(f"[Instance {self._instance}] {err_msg}")

        # MANUAL ERROR STATE CHECK
        if result.get("state") == "Error":
            err_msg = result.get("message", "Unknown Go error on CreateNode")
            logger.error(f"[Instance {self._instance}] Go error: {err_msg}")
            raise P2PError(f"[Instance {self._instance}] Failed to create node: {err_msg}")

        # MANUAL DATA EXTRACTION - no type hints!
        message_data = result.get("message")
        initial_addresses = message_data.get("addresses", [])
        self._is_public = message_data.get("isPublic", False)

        # ... rest of method ...
```

### After (With TypedGoWrapper)

```python
@model_validator(mode="after")
def start_lib(self) -> Self:
    """Initializes and starts a new libp2p node."""

    # ... validation code ...

    logger.info(f"🐍 Creating Node (Instance ID: {self._instance})...")
    try:
        # SINGLE TYPED CALL
        response = P2P._wrapper.create_node(
            instance=self._instance,
            node_config=self.go_p2p_config.model_dump()
        )

        # TYPE-SAFE UNWRAP - raises ValueError on error/empty
        node_config = response.unwrap()  # Type: NodeConfigResult

        # FULL IDE AUTOCOMPLETE - no .get() needed!
        initial_addresses = node_config.addresses  # Type: List[str]
        self._is_public = node_config.is_public  # Type: bool

        # ... rest of method ...
```

**Lines of code:** 18 → 11 (39% reduction)
**Type safety:** None → Full
**IDE support:** None → Complete autocomplete

---

## Example: `connect_to` method

### Before

```python
def connect_to(self, multiaddrs: list[str]) -> Dict[str, Any]:
    # ... validation ...

    try:
        result_ptr = P2P.libp2p.ConnectTo(
            P2P._type_interface.to_go_int(self._instance),
            P2P._type_interface.to_go_json(multiaddrs),
        )
        result: dict[str, Any] = P2P._type_interface.from_go_ptr_to_json(result_ptr)
    except Exception as e:
        logger.error(f"❌ Connection to {dest_peer_id} failed: {e}")
        raise P2PError(f"Connection to {dest_peer_id} failed") from e

    if result is None:
        logger.error("Failed to connect to peer, received null result.")
        raise P2PError("Failed to connect to peer, received null result.")

    if result.get("state") == "Error":
        logger.error(f"Failed to connect: {result.get('message', 'Unknown error')}")
        raise P2PError(f"Failed to connect: {result.get('message', 'Unknown error')}")

    peer_info = result.get("message", None)
    if not peer_info:
        logger.error("Failed to connect to peer, received empty peer info.")
        raise P2PError("Failed to connect to peer, received empty peer info.")

    logger.info(f"✅ Connection initiated to peer: {peer_info.get('ID', dest_peer_id)}")
    return peer_info
```

### After

```python
def connect_to(self, multiaddrs: list[str]) -> Dict[str, Any]:
    # ... validation ...

    try:
        response = P2P._wrapper.connect_to(
            instance=self._instance,
            multiaddrs=multiaddrs
        )
        peer_info = response.unwrap()  # Type: PeerAddrInfo - raises on error

        logger.info(f"✅ Connection initiated to peer: {peer_info.id}")
        return peer_info.model_dump()  # Convert back to dict for backward compatibility

    except ValueError as e:
        logger.error(f"❌ Connection to {dest_peer_id} failed: {e}")
        raise P2PError(f"Connection to {dest_peer_id} failed") from e
```

**Lines of code:** 25 → 13 (48% reduction)
**Type safety:** `dict[str, Any]` → `PeerAddrInfo`
**Error handling:** Manual → Automatic via `.unwrap()`

---

## Example: `pop_messages` method

### Before

```python
def pop_messages(self) -> List[bytes]:
    logger.debug(f"[Instance {self._instance}] Popping message(s)...")
    try:
        go_instance_c = P2P._type_interface.to_go_int(self._instance)
        result_ptr = P2P.libp2p.PopMessages(go_instance_c)
        raw_result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

        if raw_result is None:
            logger.error(f"[Instance {self._instance}] Received null result")
            raise P2PError(f"[Instance {self._instance}] Failed to get valid JSON")

        if isinstance(raw_result, dict):
            state = raw_result.get("state")
            if state == "Empty":
                logger.debug(f"[Instance {self._instance}] Queue is empty.")
                return []
            if state == "Error":
                error_message = raw_result.get("message", "Unknown Go error")
                logger.error(f"[Instance {self._instance}] {error_message}")
                raise P2PError(f"[Instance {self._instance}] {error_message}")

            logger.warning(f"[Instance {self._instance}] Unexpected dict format")
            raise P2PError(f"[Instance {self._instance}] Unexpected dict format")

        if not isinstance(raw_result, list):
            logger.error(f"[Instance {self._instance}] Expected list")
            raise P2PError(f"[Instance {self._instance}] Expected list")

        return raw_result
```

### After

```python
def pop_messages(self) -> List[bytes]:
    logger.debug(f"[Instance {self._instance}] Popping message(s)...")
    try:
        response = P2P._wrapper.pop_messages(instance=self._instance)

        if response.is_empty:
            logger.debug(f"[Instance {self._instance}] Queue is empty.")
            return []

        messages = response.unwrap()  # Type: List[IncomingMessage]
        return messages  # Each IncomingMessage has .data as bytes

    except ValueError as e:
        logger.error(f"[Instance {self._instance}] {e}")
        raise P2PError(f"[Instance {self._instance}] {e}") from e
```

**Lines of code:** 28 → 13 (54% reduction)
**Type safety:** `List[bytes]` → `List[IncomingMessage]`
**Handles empty state:** Manual → Built-in with `.is_empty`

---

## Migration Strategy

### Step 1: Add wrapper initialization (already done in `setup_library`)

```python
cls._wrapper = TypedGoWrapper(
    libp2p=cls.libp2p,
    type_interface=cls._type_interface
)
```

### Step 2: Refactor methods one-by-one

For each method that calls `libp2p`:

1. Replace the two-step pattern with `P2P._wrapper.method_name()`
2. Use `.unwrap()` for error handling
3. Update return type hints to use Pydantic models
4. Remove manual null checks and state checks
5. Test thoroughly

### Step 3: Gradual migration

You can migrate gradually:
- Keep old methods working
- Add new typed methods alongside
- Deprecate old methods over time

### Example: Dual Interface

```python
# Old method (deprecated)
def connect_to_legacy(self, multiaddrs: list[str]) -> Dict[str, Any]:
    """Legacy method - use connect_to() instead."""
    warnings.warn("Use connect_to() instead", DeprecationWarning)
    # ... old implementation ...

# New method (typed)
def connect_to(self, multiaddrs: list[str]) -> PeerAddrInfo:
    """Connect to peer with type-safe response."""
    response = P2P._wrapper.connect_to(self._instance, multiaddrs)
    return response.unwrap()
```

## Benefits Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines of code | ~25-30 | ~10-15 | 40-50% reduction |
| Type safety | None | Full | IDE autocomplete |
| Error handling | Manual | Automatic | Less boilerplate |
| Null checks | Manual | Automatic | Fewer bugs |
| State checks | Manual | Built-in | Cleaner code |
| Readability | Medium | High | Easier to understand |
| Maintainability | Medium | High | Easier to update |
