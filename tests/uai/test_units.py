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
# Unit tests for the rules the shared golden vectors cannot pin, either because the vector runner compares by
# subset (the dropping of undeclared keys) or because the rule is a Python-side addition (the caps on what a
# model is made to read, the fence-aware truncation, the checks done when composing).
import ast
import pathlib
import pytest
from unaiverse.uai import (UaiError, build_form, build_re_ask, check_reply, compose_form,
                                compose_media, fill_slots, gen_id, interpret_reply, normalize_label,
                                parse_message, read_answer, to_model_text, truncate_outside_fences,
                                serialize_block, validate_block)

PACKAGE = pathlib.Path(__file__).parents[2] / "src" / "unaiverse" / "uai"


def a_form(**kwargs):
    fields = [{"name": "nome", "type": "text", "label": "Nome", "required": True},
              {"name": "persone", "type": "integer", "label": "Quante persone", "min": 1, "max": 10}]
    return build_form("prenota", fields, form_id="p1", lang="it", **kwargs)


def test_package_imports_nothing_but_the_standard_library():
    # It must import in a world that has no processor, and in the browser, where torch is a stub: so every
    # import is either from the standard library or relative to this package
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) > 0, "the package directory moved: fix PACKAGE or this test checks nothing"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in ("torch", "unaiverse", "PIL", "numpy"), path.name
            elif isinstance(node, ast.ImportFrom):
                assert node.level > 0 or node.module.split(".")[0] not in ("torch", "unaiverse"), path.name


def test_check_reply_drops_undeclared_keys():
    # Reply values are attacker-controlled JSON: only what the form declares may reach the world
    values, missing, issues = check_reply(a_form(), {"nome": "Mario", "ghost": {"deep": [1, 2]}})
    assert values == {"nome": "Mario"}
    assert missing == [] and issues == {}


def test_check_reply_reports_missing_and_issues():
    values, missing, issues = check_reply(a_form(), {"persone": 25})
    assert values == {} and missing == ["nome"] and issues == {"persone": "above_max"}

    # A canonical value is type-checked, never re-parsed: the string "3" is not the number 3
    values, missing, issues = check_reply(a_form(), {"nome": "Mario", "persone": "3"})
    assert values == {"nome": "Mario"} and issues == {"persone": "type"}


def test_fill_slots_gate():
    form = a_form()

    # Complete and clean: encoded
    assert fill_slots("Nome: Mario Rossi\nQuante persone: 4", form) == {"nome": "Mario Rossi", "persone": 4}

    # A required field missing, a constraint violated, or a text that is not an answer: nothing is encoded,
    # because a partial block would stop the receiver's ladder before it reads the words
    assert fill_slots("Quante persone: 4", form) is None
    assert fill_slots("Nome: Mario\nQuante persone: 25", form) is None
    assert fill_slots("Buongiorno, come va?", form) is None

    # Already canonical: nothing to add
    already = "Nome: Mario\n\n" + serialize_block({"v": 1, "kind": "reply", "to": "p1", "values": {"nome": "M"}})
    assert fill_slots(already, form) is None


def test_asking_again_only_for_what_is_missing():
    form = a_form()

    # An answer that satisfies the form is not asked again: only "nome" was required
    values, event = read_answer("Nome: Mario", form)
    assert values == {"nome": "Mario"} and event.missing == [] and event.issues == {}

    # This one does not: nothing was said about a required field, and a value is out of range
    values, partial = read_answer("Quante persone: 25", form)
    assert values is None
    assert partial.missing == ["nome"] and partial.issues == {"persone": "above_max"}

    again = build_re_ask(form, only=partial.missing + list(partial.issues))
    assert [f["name"] for f in again["fields"]] == ["nome", "persone"]
    assert again["id"] != form["id"], "a new question, or an answer to one would answer the other"
    assert again["lang"] == form["lang"]

    # A form left with one field is answered by a bare value, which is the easiest thing to write
    one = build_re_ask(form, only=["persone"])
    assert interpret_reply("4", pending=[one]).values == {"persone": 4}

    # There has to be something to ask
    with pytest.raises(UaiError) as err:
        build_re_ask(form, only=["nothing_like_this"])
    assert err.value.reason == "no_fields"


def test_normalize_label_does_not_fold_diacritics():
    # The receiver folds neither accents nor case beyond lowercasing: composing must agree, or a form the
    # web application accepts would be refused here (and the other way round)
    assert normalize_label("  Città   di  Nascita ") == "città di nascita"
    assert normalize_label("Si") != normalize_label("Sì")


def test_compose_round_trips_and_refuses_invalid_blocks():
    text = compose_form("prenota", [{"name": "nome", "type": "text", "label": "Nome"}], lang="it")
    parts = parse_message(text)
    assert len(parts) == 1 and parts[0]["type"] == "form" and not parts[0].get("degraded")

    media = compose_media("https://cdn.unaiverse.io/a.png", alt="[foto] Un gatto")
    assert validate_block(parse_message(media)[0]["spec"])["ok"]

    # Only https travels, and a comma in an option label would break the positional rung of the receiver
    with pytest.raises(UaiError):
        compose_media("http://cdn.unaiverse.io/a.png", alt="[foto]")

    # A URL that does not say what it carries: valid for the protocol, undrawable for a renderer, so it is
    # refused here rather than shipped as a generic binary blob
    with pytest.raises(UaiError) as err:
        compose_media("https://cdn.unaiverse.io/image/1234", alt="[foto]")
    assert err.value.reason == "mime_unknown"
    assert compose_media("https://cdn.unaiverse.io/image/1234", alt="[foto]", mime="image/png")
    with pytest.raises(UaiError) as err:
        build_form("f", [{"name": "a", "type": "select",
                          "options": [{"value": "x", "label": "uno, due"}]}])
    assert err.value.reason == "option_label_comma:a"

    # Two labels that normalise the same are a duplicate for the receiver too
    with pytest.raises(UaiError) as err:
        build_form("f", [{"name": "a", "type": "select",
                          "options": [{"value": "x", "label": "Anna"}, {"value": "y", "label": "anna "}]}])
    assert err.value.reason == "option_label_duplicate:a"


def test_generated_instruction_is_capped():
    # An instruction longer than the receiver accepts makes the whole block degrade to text, so a widget would
    # silently become prose: composing refuses instead
    options = [{"value": f"v{i}", "label": f"Opzione numero {i} " + "x" * 100} for i in range(40)]
    with pytest.raises(UaiError) as err:
        build_form("grande", [{"name": "scelta", "type": "select", "options": options}])
    assert err.value.reason == "alt_too_long"


def test_gen_id_stays_within_the_alphabet_of_the_contract():
    plain = gen_id()
    assert len(plain) == 6 and plain.isalnum()
    tagged = gen_id("Anna Bianchi (stanza 3)")
    assert len(tagged) <= 32 and all(c.isalnum() or c in "_-" for c in tagged)
    assert gen_id("!!!").isalnum()


def test_model_view_is_bounded():
    block = compose_media("https://cdn.unaiverse.io/a.png", alt="[foto]")
    many = "\n".join([block] * 30)
    view = to_model_text(many, max_blocks=5)
    assert view.count("[foto]") == 5 and "25 more blocks omitted" in view
    assert to_model_text(many, max_blocks=100, max_bytes=64).endswith("[... truncated]")


def test_truncation_never_leaves_half_a_fence():
    block = compose_form("prenota", [{"name": "nome", "type": "text", "label": "Nome"}], lang="it")
    text = "Ciao, ecco il modulo.\n\n" + block

    # A cut that would fall inside the block drops the whole block
    cut = truncate_outside_fences(text, len(text) - 40)
    assert "```" not in cut and cut == "Ciao, ecco il modulo."
    assert all(not p.get("degraded") for p in parse_message(cut))

    # A cut that falls outside it changes nothing, and a short text is returned untouched
    assert truncate_outside_fences(text, len(text)) == text
    assert truncate_outside_fences("breve", 100) == "breve"
