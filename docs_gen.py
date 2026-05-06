from pathlib import Path
from collections import defaultdict
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent / "src"



packages: dict[tuple, dict] = {}            
all_module_parts: list[tuple] = []         

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    parts = tuple(module_path.parts)

    if parts[-1] == "__main__":
        continue

    if parts[-1] == "__init__":
        pkg_parts = parts[:-1]
        if pkg_parts not in packages:
            packages[pkg_parts] = {"modules": [], "subpackages": []}
    else:
        all_module_parts.append(parts)

for parts in all_module_parts:
    parent = parts[:-1]
    if parent in packages:
        packages[parent]["modules"].append(parts)

for pkg_parts in list(packages.keys()):
    parent = pkg_parts[:-1]
    if parent in packages:
        if pkg_parts not in packages[parent]["subpackages"]:
            packages[parent]["subpackages"].append(pkg_parts)

PACKAGE_DESCRIPTIONS = {
    ("unaiverse",): (
        "Root package of the UNaIVERSE framework — "
        "a Collectionless AI system for autonomous agents."
    ),
    ("unaiverse", "hsm"): (
        "Hybrid state machine: actions, states, policites."
    ),
    ("unaiverse", "modules"): (
        "Neural-network modules: CNU layers, high-level utilities, "
        "and shared network building blocks."
    ),
    ("unaiverse", "modules", "cnu"): (
        "**Continuous Neural Unit (CNU)** — custom layers, psi functions, "
        "and CNU network definitions."
    ),
    ("unaiverse", "modules", "hl"): (
        "High-level learning utilities and helpers."
    ),
    ("unaiverse", "networking"): (
        "Peer-to-peer networking stack: node management, connection pools, "
        "and the libp2p transport layer."
    ),
    ("unaiverse", "networking", "node"): (
        "Node-level abstractions: connection pools, profiles, "
        "authentication tokens, and the main Node class."
    ),
    ("unaiverse", "networking", "p2p"): (
        "Low-level P2P transport built on Go-libp2p: "
        "message serialisation, protocol types, and the P2P driver."
    ),
    ("unaiverse", "streams"): (
        "Stream library internals for data-stream processing."
    ),
    ("unaiverse", "utils"): (
        "Miscellaneous utilities, sandboxing, and helper functions."
    ),
}

MODULE_DESCRIPTIONS = {
    ("unaiverse", "agent"): "Main Agent class implementing autonomous agent behavior and lifecycle management.",
    ("unaiverse", "agent_basics"): "Foundational agent components and basic agent functionality.",
    ("unaiverse", "clock"): "Timing and clock management for agent synchronization.",
    ("unaiverse", "custom"): "Global options (some of them loaded using environment variables).",
    ("unaiverse", "interaction"): "Interaction management.",
    ("unaiverse", "stats"): "Statistical tracking and metrics collection for agents.",
    ("unaiverse", "stats_html_renderer"): "Converter to dump stat metrics to HTML.",
    ("unaiverse", "world"): "World simulation environment and world state management.",

    ("unaiverse", "hsm", "action"): "Action in a Hybrid State Machine (HSM).",
    ("unaiverse", "hsm", "has"): "Hybrid State Machine (HSM) implementation for agent state management.",
    ("unaiverse", "hsm", "state"): "State in a Hybrid State Machine (HSM).",
    
    ("unaiverse", "networking", "node", "node"): (
        "Core Node class providing the main interface for peer-to-peer networking, "
        "agent management, and world coordination."
    ),
    ("unaiverse", "networking", "node", "profile"): (
        "NodeProfile class managing static and dynamic node information, "
        "including system specs, CV, location, and connection metadata."
    ),
    ("unaiverse", "networking", "node", "connpool"): (
        "ConnectionPools and NodeConn classes for managing multiple connection pools "
        "across public and private P2P networks with token-based authentication."
    ),
    ("unaiverse", "networking", "node", "tokens"): (
        "TokenVerifier class for JWT-based authentication and authorization "
        "using RS256 public key cryptography."
    ),
    
    ("unaiverse", "networking", "p2p", "p2p"): (
        "Main P2P class wrapping Go-libp2p functionality for peer connections, "
        "messaging, and topic-based pub/sub communication."
    ),
    ("unaiverse", "networking", "p2p", "messages"): (
        "Message serialization and deserialization using Protocol Buffers "
        "for efficient network communication."
    ),
    ("unaiverse", "networking", "p2p", "lib_types"): (
        "Type definitions and data structures for P2P library integration "
        "with Go-libp2p backend."
    ),
    ("unaiverse", "networking", "p2p", "mylogger"): (
        "Custom logging utilities for P2P networking operations and debugging."
    ),
    ("unaiverse", "networking", "p2p", "golibp2p"): (
        "Python bindings and interface to the Go-libp2p shared library."
    ),
    
    ("unaiverse", "modules", "networks"): (
        "Neural network architectures and model definitions using CNU layers."
    ),
    ("unaiverse", "modules", "utils"): (
        "Utility functions for neural network training, evaluation, and data processing."
    ),
    ("unaiverse", "streams", "dataprops"): (
        "Data properties and structured data handling for streams."
    ),
    ("unaiverse", "streams", "streamlib"): (
        "Pre-designed streams of different types."
    ),
    ("unaiverse", "streams", "streamproxy"): (
        "A proxy object that binds multiple streams."
    ),
    ("unaiverse", "streams", "streams"): (
        "Streams and Buffered Streams."
    ),
    ("unaiverse", "utils", "logger"): (
        "Main logger class, used in the whole project."
    ),
    ("unaiverse", "utils", "os_fd_capture"): (
        "Helper for the main logger class, used to capture external loggings."
    ),
    ("unaiverse", "utils", "sandbox"): (
        "Docker-based sandboxing utilities for running UNaIVERSE nodes in isolated containers "
        "with configurable read-only and writable mount paths."
    ),
    ("unaiverse", "utils", "misc"): (
        "Miscellaneous helper functions including file tracking, node address management, "
        "JSON monitoring, key management, policy filters, and countdown utilities."
    ),
}


def _label(parts: tuple) -> str:
    """Last component with a leading capital, e.g. ('unaiverse','agent') → 'agent'."""
    return parts[-1]



for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__main__":
        continue

    if parts[-1] == "__init__":
        pkg_parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        nav[pkg_parts] = doc_path.as_posix()

        pkg_info = packages.get(pkg_parts, {"modules": [], "subpackages": []})
        pkg_name = ".".join(pkg_parts)
        description = PACKAGE_DESCRIPTIONS.get(pkg_parts, "")

        lines = []
        lines.append(f"# `{pkg_name}`\n")
        if description:
            lines.append(f"{description}\n")
        lines.append("---\n")

        if pkg_info["subpackages"]:
            lines.append("## Sub-packages\n")
            lines.append("| Package | Description |")
            lines.append("|---------|-------------|")
            for sp in sorted(pkg_info["subpackages"]):
                sp_name = ".".join(sp)
                sp_link = "/".join(sp[len(pkg_parts):]) + "/index.md"
                sp_desc = PACKAGE_DESCRIPTIONS.get(sp, "")
                short = sp_desc.split(".")[0] + "." if sp_desc else ""
                lines.append(f"| [`{sp_name}`]({sp_link}) | {short} |")
            lines.append("")

        if pkg_info["modules"]:
            lines.append("## Modules\n")
            lines.append("| Module | Description |")
            lines.append("|--------|-------------|")
            for mod in sorted(pkg_info["modules"]):
                mod_name = ".".join(mod)
                mod_link = mod[-1] + ".md"
                mod_desc = MODULE_DESCRIPTIONS.get(mod, "")
                lines.append(f"| [`{mod_name}`]({mod_link}) | {mod_desc} |")
            lines.append("")

        content = "\n".join(lines)

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            fd.write(content)
    else:
        # ── Regular module page ───────────────────────────────────
        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(doc_path, path)

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())