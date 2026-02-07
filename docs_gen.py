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
    ("unaiverse", "streamlib"): (
        "Stream library internals for data-stream processing."
    ),
    ("unaiverse", "utils"): (
        "Miscellaneous utilities, sandboxing, and helper functions."
    ),
}


def _label(parts: tuple) -> str:
    """Last component with a leading capital, e.g. ('unaiverse','agent') → 'agent'."""
    return parts[-1]


# ── Second pass: generate docs ────────────────────────────────────────

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__main__":
        continue

    if parts[-1] == "__init__":
        # ── Package index page ────────────────────────────────────
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

        # List sub-packages
        if pkg_info["subpackages"]:
            lines.append("## Sub-packages\n")
            lines.append("| Package | Description |")
            lines.append("|---------|-------------|")
            for sp in sorted(pkg_info["subpackages"]):
                sp_name = ".".join(sp)
                sp_link = "/".join(sp[len(pkg_parts):]) + "/index.md"
                sp_desc = PACKAGE_DESCRIPTIONS.get(sp, "")
                # Trim to first sentence for the table
                short = sp_desc.split(".")[0] + "." if sp_desc else ""
                lines.append(f"| [`{sp_name}`]({sp_link}) | {short} |")
            lines.append("")

        # List modules
        if pkg_info["modules"]:
            lines.append("## Modules\n")
            lines.append("| Module | Description |")
            lines.append("|--------|-------------|")
            for mod in sorted(pkg_info["modules"]):
                mod_name = ".".join(mod)
                mod_link = mod[-1] + ".md"
                lines.append(f"| [`{mod_name}`]({mod_link}) | |")
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