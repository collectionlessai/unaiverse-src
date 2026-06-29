"""Characterization tests for the pure helpers and value types in
``unaiverse.streams.dataprops`` (net-hash statics, StreamType, DataProps).

These tests pin the CURRENT behavior of the module. Where the behavior looks
suspect (see the ``# SUSPECTED BUG`` notes), the test documents what the code
does today rather than what it arguably should do.
"""
import torch
import pytest

from unaiverse.streams.dataprops import DataProps, StreamType

pytestmark = pytest.mark.unit


# ── net_hash statics ─────────────────────────────────────────────────────────

def test_build_net_hash_direct_message_form():
    """build_net_hash with pubsub=False uses the ``::dm:`` separator."""
    assert DataProps.build_net_hash("peerA", False, "foo") == "peerA::dm:foo"


def test_build_net_hash_pubsub_form():
    """build_net_hash with pubsub=True uses the ``::ps:`` separator."""
    assert DataProps.build_net_hash("peerA", True, "foo") == "peerA::ps:foo"


def test_is_pubsub_from_net_hash_discriminates_forms():
    """is_pubsub_from_net_hash is True only for the ``::ps:`` form."""
    ps = DataProps.build_net_hash("peerA", True, "foo")
    dm = DataProps.build_net_hash("peerA", False, "foo")
    assert DataProps.is_pubsub_from_net_hash(ps) is True
    assert DataProps.is_pubsub_from_net_hash(dm) is False


def test_peer_id_from_net_hash_extracts_prefix():
    """peer_id_from_net_hash returns the part before the first ``::``."""
    assert DataProps.peer_id_from_net_hash("peerA::dm:foo") == "peerA"
    assert DataProps.peer_id_from_net_hash("peerA::ps:topic") == "peerA"


def test_user_hash_from_net_hash_joins_prefix_and_name():
    """user_hash_from_net_hash keeps the peer prefix and appends ``:name``."""
    h = DataProps.user_hash_from_net_hash("peerA::dm:foo", "myname")
    assert h == "peerA:myname"
    # The name argument is independent of the name_or_group baked into the hash.
    assert DataProps.peer_id_from_user_hash(h) == "peerA"
    assert DataProps.name_from_user_hash(h) == "myname"


def test_normalize_net_hash_strips_role_tilde_suffix():
    """normalize_net_hash (dm form, with ``~``) keeps the prefix and replaces the
    body with whatever follows the first ``~`` (the role suffix)."""
    out = DataProps.normalize_net_hash("abc::dm:0-foo~bar")
    assert out == "abc::dm:bar"


def test_normalize_net_hash_strips_id_mapping_dash():
    """normalize_net_hash (dm form, no ``~``) drops the leading ``<id>-`` mapping,
    keeping the part after the first dash."""
    out = DataProps.normalize_net_hash("abc::dm:7-foo")
    assert out == "abc::dm:foo"


def test_normalize_net_hash_passthrough_for_pubsub():
    """normalize_net_hash leaves a pubsub (``::ps:``) hash untouched."""
    h = "abc::ps:7-foo"
    assert DataProps.normalize_net_hash(h) == h


def test_name_or_group_from_net_hash_for_both_forms():
    """name_or_group_from_net_hash returns the body after the proper separator."""
    assert DataProps.name_or_group_from_net_hash("abc::dm:foo") == "foo"
    assert DataProps.name_or_group_from_net_hash("abc::ps:topic") == "topic"


# ── DataProps net-hash instance helpers ──────────────────────────────────────

def test_instance_net_hash_uses_name_when_group_none():
    """When group is 'none', net_hash falls back to the stream name (dm form)."""
    dp = DataProps(name="foo", group="none", data_type="text", pubsub=False)
    assert dp.name_or_group() == "foo"
    assert dp.net_hash("peerA") == "peerA::dm:foo"


def test_instance_net_hash_uses_group_when_set_and_pubsub():
    """When group is set it wins over name; pubsub=True selects the ps form."""
    dp = DataProps(name="foo", group="g1", data_type="text", pubsub=True)
    assert dp.name_or_group() == "g1"
    assert dp.net_hash("peerA") == "peerA::ps:g1"


def test_instance_is_pubsub_reflects_constructor_flag():
    """is_pubsub returns the stored pubsub flag."""
    assert DataProps(data_type="text", pubsub=True).is_pubsub() is True
    assert DataProps(data_type="text", pubsub=False).is_pubsub() is False


def test_instance_user_hash_uses_name_not_group():
    """user_hash always uses the bare name, never the group."""
    dp = DataProps(name="foo", group="g1", data_type="text")
    assert dp.user_hash("peerA") == "peerA:foo"


# ── StreamType ───────────────────────────────────────────────────────────────

def test_streamtype_shorthand_sets_data_type():
    """StreamType('text') is shorthand for data_type='text' on each DataProps."""
    st = StreamType("text")
    for d in st.to_list_of_dicts():
        assert d["data_type"] == "text"


def test_streamtype_default_emits_private_and_public():
    """A plain StreamType emits two DataProps: one private then one public."""
    st = StreamType("text")
    dicts = st.to_list_of_dicts()
    assert len(dicts) == 2
    assert [d["public"] for d in dicts] == [False, True]


def test_streamtype_private_only_emits_single_private():
    """private_only=True emits exactly one (private) DataProps."""
    st = StreamType("text", private_only=True)
    dicts = st.to_list_of_dicts()
    assert len(dicts) == 1
    assert dicts[0]["public"] is False


def test_streamtype_rejects_public_kwarg():
    """Passing 'public' directly (a DataProps arg) raises ValueError."""
    with pytest.raises(ValueError):
        StreamType("text", public=True)


def test_streamtype_to_dict_is_unsupported():
    """to_dict on the container raises (it is a per-DataProps method)."""
    st = StreamType("text")
    with pytest.raises(RuntimeError):
        st.to_dict()


def test_streamtype_clone_is_distinct_but_equal():
    """clone returns a new StreamType with cloned, content-equal DataProps."""
    st = StreamType("text")
    cl = st.clone()
    assert cl is not st
    assert cl.props is not st.props
    # Same number of props and identical serialized content.
    assert st.to_list_of_dicts() == cl.to_list_of_dicts()
    # Each contained DataProps is a fresh object.
    for a, b in zip(st.props, cl.props):
        assert a is not b


def test_streamtype_getattr_delegates_to_first_prop():
    """A non set_* method not defined on StreamType delegates to the first
    (private) DataProps. (is_public is shadowed by StreamType and raises.)"""
    st = StreamType("text")
    assert st.is_text() is True            # delegated to props[0]
    assert st.get_name() == "unk"          # delegated to props[0]
    with pytest.raises(RuntimeError):
        st.is_public()                     # explicitly overridden on StreamType


def test_streamtype_set_method_applies_to_all_props():
    """A set_* call is broadcast to every contained DataProps."""
    st = StreamType("text")
    st.set_name("renamed")
    for p in st.props:
        assert p.get_name() == "renamed"


# ── DataProps: round-trip and clone ──────────────────────────────────────────

def test_dataprops_to_from_dict_roundtrip_text():
    """to_dict -> from_dict round-trips a plain text DataProps."""
    dp = DataProps(name="n", group="g", data_type="text", data_desc="d",
                   delta=0.5, pubsub=True, public=True)
    d = dp.to_dict()
    dp2 = DataProps.from_dict(d)
    assert dp2.to_dict() == d


def test_dataprops_to_from_dict_roundtrip_tensor_with_labels():
    """to_dict -> from_dict round-trips a tensor DataProps with labels/dtype."""
    dp = DataProps(name="t", data_type="tensor", tensor_shape=(1, 2),
                   tensor_dtype="torch.float32", tensor_labels=["a", "b"])
    d = dp.to_dict()
    # Serialized fields are simple/JSON-friendly.
    assert d["tensor_dtype"] == "torch.float32"
    assert d["tensor_shape"] == (1, 2)
    assert d["tensor_labels"] == {"labels": ["a", "b"], "labeling_rule": "max"}
    dp2 = DataProps.from_dict(d)
    assert dp2.to_dict() == d


def test_dataprops_to_dict_keys_are_stable():
    """to_dict exposes exactly the documented serialization keys."""
    dp = DataProps(data_type="text")
    assert set(dp.to_dict().keys()) == {
        "name", "group", "data_type", "data_desc",
        "tensor_shape", "tensor_dtype", "tensor_labels",
        "delta", "pubsub", "public",
    }


def test_dataprops_clone_is_distinct_and_scalar_independent():
    """clone returns a distinct, content-equal object whose scalar fields can be
    mutated without touching the original."""
    dp = DataProps(name="orig", group="g0", data_type="text", delta=1.0)
    cl = dp.clone()
    assert cl is not dp
    assert cl.to_dict() == dp.to_dict()
    cl.set_name("changed")
    cl.set_group("g1")
    cl.set_delta(9.0)
    cl.set_public(True)
    cl.set_pubsub(True)
    assert dp.get_name() == "orig"
    assert dp.get_group() == "g0"
    assert dp.delta == 1.0
    assert dp.is_public() is False
    assert dp.is_pubsub() is False


def test_dataprops_clone_copies_label_list():
    """clone copies the labels list, so clone and original do not share one
    underlying list and mutating one does not affect the other."""
    dp = DataProps(data_type="tensor", tensor_shape=(1, 2),
                   tensor_dtype="torch.float32", tensor_labels=["a", "b"])
    cl = dp.clone()
    assert cl.tensor_labels.labels is not dp.tensor_labels.labels
    assert cl.tensor_labels.labels == dp.tensor_labels.labels
    cl.tensor_labels.labels.append("c")
    assert dp.tensor_labels.labels == ["a", "b"]


# ── DataProps: type predicates ───────────────────────────────────────────────

def test_predicate_text():
    """is_text is True only for data_type='text'."""
    dp = DataProps(data_type="text")
    assert dp.is_text() is True
    assert dp.is_tensor() is False
    assert dp.is_img() is False
    assert dp.is_file() is False
    assert dp.is_all() is False


def test_predicate_img():
    """is_img is True only for data_type='img'."""
    dp = DataProps(data_type="img")
    assert dp.is_img() is True
    assert dp.is_text() is False
    assert dp.is_tensor() is False


def test_predicate_file():
    """is_file is True only for data_type='file'."""
    dp = DataProps(data_type="file")
    assert dp.is_file() is True
    assert dp.is_text() is False


def test_predicate_all():
    """is_all is True only for the wildcard data_type='all'."""
    dp = DataProps(data_type="all")
    assert dp.is_all() is True
    assert dp.is_text() is False
    assert dp.is_tensor() is False


def test_predicate_tensor_and_dtype_helpers():
    """is_tensor plus the dtype helpers reflect a float tensor's properties."""
    dp = DataProps(data_type="tensor", tensor_shape=(1, 3, 8, 8),
                   tensor_dtype="torch.float32")
    assert dp.is_tensor() is True
    assert dp.is_tensor_float() is True
    assert dp.is_tensor_long() is False
    # 4D with 3 channels reads as an image tensor.
    assert dp.is_tensor_img() is True


def test_predicate_tensor_token_ids_and_target_id():
    """Long dtype shapes drive the token-id / target-id predicates."""
    token_ids = DataProps(data_type="tensor", tensor_shape=(1, None),
                          tensor_dtype="torch.long")
    assert token_ids.is_tensor_long() is True
    assert token_ids.is_tensor_token_ids() is True
    assert token_ids.is_tensor_target_id() is False

    target_id = DataProps(data_type="tensor", tensor_shape=(None,),
                          tensor_dtype="torch.long")
    assert target_id.is_tensor_target_id() is True
    assert target_id.is_tensor_token_ids() is False


def test_invalid_data_type_raises():
    """An unknown data_type is rejected by the constructor assertion."""
    with pytest.raises(AssertionError):
        DataProps(data_type="bogus")


def test_tilde_in_name_rejected():
    """A '~' in the stream name is rejected (it is a hash separator char)."""
    with pytest.raises(AssertionError):
        DataProps(data_type="text", name="bad~name")
