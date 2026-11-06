"""Tests for ``prune_to_import_closure``: the shipped-code perimeter helper.

The closure mirrors the in-memory loader rules on the joiner side: relative
imports resolve against the shipped set, absolute imports never do, the root
``__init__.py`` is unreachable, and forced members (stats.py) are included
without following their own imports.
"""

import pytest

from unaiverse.utils.misc import prune_to_import_closure

pytestmark = pytest.mark.unit


def test_basic_closure_keeps_roles_and_transitive_helpers():
    files = {
        'guest.py': "from .config import Config\nfrom .utils import fmt\n",
        'utils.py': "from .config import Config\n",
        'config.py': "X = 1\n",
        'world.py': "from .stats import WStats\nfrom .guest import WAgent\n",
        'html_renderer.py': "def render():\n    pass\n",
    }
    pruned = prune_to_import_closure(files, ['guest.py'])
    assert set(pruned) == {'guest.py', 'utils.py', 'config.py'}


def test_forced_member_included_without_following_its_imports():
    files = {
        'role.py': "X = 1\n",
        'stats.py': "def plot(self):\n    from ._vault import DATA\n    return DATA\n",
        '_vault.py': "DATA = 'secret'\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'], forced=('stats.py',))
    assert set(pruned) == {'role.py', 'stats.py'}
    assert '_vault.py' not in pruned


def test_forced_member_is_traversed_when_a_role_imports_it():
    """A role importing stats turns it into a real namespace module on the joiner,
    so its relative imports resolve there and its dependencies must ship too."""
    files = {
        'role.py': "from .stats import WStats\n",
        'stats.py': "from .helper import H\nclass WStats:\n    pass\n",
        'helper.py': "H = 1\n",
        '_vault.py': "DATA = 'secret'\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'], forced=('stats.py',))
    assert set(pruned) == {'role.py', 'stats.py', 'helper.py'}


def test_namespace_package_import_keeps_whole_subtree():
    """The loader auto-creates namespace packages for bare directories, so
    'from . import pkg' resolves at runtime: the whole subtree must ship."""
    files = {
        'role.py': "from . import pkg\n",
        'pkg/inner.py': "i = 1\n",
        'pkg/deep/leaf.py': "leaf = 1\n",
        'other.py': "o = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py', 'pkg/inner.py', 'pkg/deep/leaf.py'}


def test_namespace_submodule_alias_keeps_only_that_module():
    files = {
        'role.py': "from .pkg import mod\n",
        'pkg/mod.py': "m = 1\n",
        'pkg/unrelated.py': "u = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py', 'pkg/mod.py'}


def test_star_import_on_package_keeps_whole_subtree():
    """__all__ can name submodules the AST never sees imported: conservative subtree."""
    files = {
        'role.py': "from .pkg import *\n",
        'pkg/__init__.py': "__all__ = ['sub1']\n",
        'pkg/sub1.py': "s = 1\n",
        'pkg/sub2.py': "t = 2\n",
        'other.py': "o = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py', 'pkg/__init__.py', 'pkg/sub1.py', 'pkg/sub2.py'}


def test_missing_forced_and_missing_entry_are_ignored():
    files = {'role.py': "X = 1\n"}
    pruned = prune_to_import_closure(files, ['role.py', 'ghost.py'], forced=('stats.py',))
    assert set(pruned) == {'role.py'}


def test_root_init_is_never_shipped_even_if_it_imports_everything():
    files = {
        '__init__.py': "from . import world\nfrom . import role\n",
        'role.py': "X = 1\n",
        'world.py': "Y = 2\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py'}


def test_subpackage_module_brings_its_init_along():
    files = {
        'role.py': "from .lib.helper import h\n",
        'lib/__init__.py': "",
        'lib/helper.py': "from .other import o\n",
        'lib/other.py': "o = 1\n",
        'lib/unused.py': "z = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py', 'lib/__init__.py', 'lib/helper.py', 'lib/other.py'}


def test_from_dot_import_module_form():
    files = {
        'role.py': "from . import util_a, util_b\n",
        'util_a.py': "a = 1\n",
        'util_b.py': "b = 2\n",
        'util_c.py': "c = 3\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py', 'util_a.py', 'util_b.py'}


def test_absolute_imports_are_not_followed():
    files = {
        'role.py': "import utils\nimport json\nfrom os import path\n",
        'utils.py': "u = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py'}


def test_lazy_function_body_imports_are_followed():
    files = {
        'role.py': "def act():\n    from .late import thing\n    return thing\n",
        'late.py': "thing = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py', 'late.py'}


def test_relative_import_escaping_the_tree_is_ignored():
    files = {
        'role.py': "from ..outside import x\n",
        'other.py': "x = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py'}


def test_unparsable_reached_file_is_kept_but_contributes_no_edges():
    files = {
        'role.py': "from .broken import x\n",
        'broken.py': "def broken(:\n",
        'hidden.py': "x = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role.py'])
    assert set(pruned) == {'role.py', 'broken.py'}


def test_multiple_entry_points_union_and_order_preserved():
    files = {
        'a_helper.py': "h = 1\n",
        'role1.py': "from .a_helper import h\n",
        'role2.py': "from .z_helper import z\n",
        'world.py': "w = 1\n",
        'z_helper.py': "z = 1\n",
    }
    pruned = prune_to_import_closure(files, ['role1.py', 'role2.py'])
    assert set(pruned) == {'role1.py', 'role2.py', 'a_helper.py', 'z_helper.py'}
    assert list(pruned) == [k for k in files if k in pruned]


def test_turing_hotel_shape():
    """Mirror of the maintained example: roles + domain objects ship, world-side stays."""
    files = {
        'config.py': "import textwrap\n",
        'floor_manager.py': "from .floor import Floor\nfrom .config import Config\nfrom .utils import fmt\n",
        'floor.py': "from .room import Room\nfrom .config import Config\nfrom .utils import fmt\n",
        'guest.py': "from .config import Config\nfrom .utils import fmt\n",
        'hotel_manager.py': "from .room import Room\nfrom .hotel import Hotel\nfrom .config import Config\n",
        'hotel.py': "from .room import Room\nfrom .floor import Floor\nfrom .config import Config\n",
        'html_renderer.py': "def render():\n    pass\n",
        '__init__.py': "",
        'room.py': "from .config import Config\n",
        'stats.py': "def plot():\n    from .html_renderer import render\n    return render\n",
        'utils.py': "from .config import Config\ndef fmt():\n    pass\n",
        'world.py': "from .stats import WStats\nfrom .config import Config\n",
    }
    roles = ['floor_manager.py', 'guest.py', 'hotel_manager.py']
    pruned = prune_to_import_closure(files, roles, forced=('stats.py',))
    assert set(pruned) == {
        'floor_manager.py', 'guest.py', 'hotel_manager.py',
        'floor.py', 'hotel.py', 'room.py', 'config.py', 'utils.py',
        'stats.py',
    }
    assert 'world.py' not in pruned and 'html_renderer.py' not in pruned
    assert '__init__.py' not in pruned
