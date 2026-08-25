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
# Best-effort filling of a form from what a processor wrote in words. This is the half that lets a model
# answer a widget: the model reads the instruction and replies in prose, and this turns the prose into the
# canonical block the asker can read at the first rung of the ladder.
#
# A block is a message with a contract on its format, so an answer that does not honour the form is not
# sent as it is: whoever wrote it is asked again, a model through a corrective prompt fed back to it within
# the same turn, a person through a line on their screen, and only what satisfies the form travels. The
# pieces of that loop live here (reading, describing what is wrong, wording the second request, encoding
# what could be read once the asking is over); the loop itself sits around the processor.
from dataclasses import dataclass
from .draft import encode_reply
from .interpret import interpret_reply
from .templates import template_for
from .constants import VIA_BLOCK, VIA_FREETEXT
from .validate import field_label, interactive_fields

# Wording of the second request, by language. Reason codes stay the closed vocabulary of the contract.
_RETRY = {
    "it": {"missing": "non hai detto nulla su {fields}",
           "issues": "questi valori non sono accettabili: {fields}",
           "nothing": "non è stato possibile leggere nessun campo",
           "again": "La tua risposta era:\n{answer}\n\nNon soddisfa la richiesta: {problem}. Rispondi di nuovo, "
                    "una riga per campo nella forma 'campo: valore', senza altro testo.",
           "person": "La risposta a '{form}' non soddisfa la richiesta ({problem}): non è stata inviata. "
                     "Scrivila di nuovo."},
    "en": {"missing": "nothing was said about {fields}",
           "issues": "these values are not acceptable: {fields}",
           "nothing": "no field of it could be read",
           "again": "Your answer was:\n{answer}\n\nIt does not satisfy the request: {problem}. Answer again, "
                    "one line per field in the form 'field: value', with no other text.",
           "person": "The answer to '{form}' does not satisfy the request ({problem}): it was not sent. "
                     "Write it again."},
}


class AnswerWithheld(Exception):
    """Raised through the processor call when what it produced must not travel.

    It is how the gate turns a turn into a silent failure: nothing reaches the output stream, the interaction
    that asked stays open, and whoever is at the keyboard has been told to answer again.
    """


@dataclass(frozen=True)
class AnswerOutcome:
    """What the gate decided about the text a processor produced in answer to a form.

    Exactly one of the three readings applies: ``retry`` carries the corrective prompt to feed the module
    again; ``withhold`` says nothing travels; otherwise ``text`` is what travels, or None to let the words
    go as they were written.
    """
    text: str | None = None
    retry: str | None = None
    withhold: bool = False


def read_answer(text, spec: dict, lang: str | None = None) -> tuple[dict | None, object]:
    """Reads a text as the answer to one form, and says both what it gives and how it was understood.

    Args:
        text: What the processor produced.
        spec: The validated spec of the form that was asked.
        lang: Fallback language, used when the spec declares none.

    Returns:
        A tuple with the canonical values, when every required field was answered and nothing violates the
        spec, or None when they must not be encoded, and the event the text was read as, or None when there
        was nothing to read at all. The event is what tells an answer that fell short from a text that was
        never an answer, which is the difference between asking again and staying quiet.
    """
    if not isinstance(text, str) or not text.strip() or not spec:
        return None, None

    event = interpret_reply(text, pending=[spec], lang=lang)

    # Already canonical (the processor wrote the block itself), or not an answer at all, or an answer to
    # something else entirely
    if event.via in (VIA_BLOCK, VIA_FREETEXT) or event.to != spec.get("id"):
        return None, event

    # The form is not satisfied: this is what the asking-again loop is for
    if event.issues or event.missing or not event.values:
        return None, event
    return event.values, event


def fill_slots(text, spec: dict, lang: str | None = None) -> dict | None:
    """The values of a clean, complete answer to one form, or None when the text does not satisfy it."""
    return read_answer(text, spec, lang=lang)[0]


def _words(lang: str | None) -> dict:
    return _RETRY.get(template_for(lang)[0], _RETRY["en"])


def describe_answer(spec: dict, event, lang: str | None = None) -> str:
    """Says, in one line, what keeps an answer from satisfying its form, naming the fields as the author did.

    Args:
        spec: The validated form that was asked.
        event: The event the answer was read as.
        lang: Language of the sentence, defaulting to the one of the form.

    Returns:
        A readable summary, for whoever wrote the answer rather than for whoever asked.
    """
    words = _words(lang if lang is not None else spec.get("lang"))
    labels = {f["name"]: field_label(f) for f in interactive_fields(spec)}
    said = []
    if getattr(event, "missing", None):
        said.append(words["missing"].format(fields=", ".join(labels.get(n, n) for n in event.missing)))
    if getattr(event, "issues", None):
        said.append(words["issues"].format(
            fields=", ".join(f"{labels.get(n, n)} ({r})" for n, r in event.issues.items())))
    if not said:
        said.append(words["nothing"])
    return "; ".join(said)


def retry_prompt(spec: dict, text: str, event, model_view: str | None = None) -> str:
    """Words the second request to a model: the instruction it was given, its answer, and what is wrong.

    Args:
        spec: The validated form that was asked.
        text: What the model wrote.
        event: The event that answer was read as.
        model_view: The rendering of the message that carried the form, as the model read it the first time.
            When it is not known any more, the instruction of the form alone is repeated.

    Returns:
        The text to feed the model in place of the original message.
    """
    words = _words(spec.get("lang"))
    view = model_view if model_view else spec.get("alt", "")
    again = words["again"].format(answer=text.strip(), problem=describe_answer(spec, event))
    return f"{view.rstrip()}\n\n{again}" if view.strip() else again


def reprompt_person(spec: dict, event) -> str:
    """Words the second request to a person at a keyboard: what was wrong, and that nothing was sent."""
    words = _words(spec.get("lang"))
    return words["person"].format(form=spec.get("name", spec.get("id")), problem=describe_answer(spec, event))


def encode_partial(spec: dict, text: str, event) -> str | None:
    """What travels when the asking is over and the answer still falls short: the words, and what they gave.

    The values that could be read are encoded beside the words, so that whoever asked reads them at the first
    rung and learns the rest is missing; the ones that violate the form are left out, as the contract wants.
    When nothing at all could be read there is nothing to add, and None says so.
    """
    values = getattr(event, "values", None) or {}
    if not values:
        return None
    return f"{text.rstrip()}\n\n{encode_reply(spec, values)}"
