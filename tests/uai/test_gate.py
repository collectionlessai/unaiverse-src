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
# Tests of the gate that sits around the processor: what a module is handed when a message carries protocol
# blocks, what its answer becomes on the way out, and what happens when that answer does not honour the form
# (a model is asked again, a person is told to write again, nothing dubious ever travels). The policy methods
# are taken from AgentBasics, so these tests exercise the code that ships, not a copy of it.
import torch
import pytest
from unaiverse.custom import Custom
import unaiverse.agent_basics as agent_basics_module
from unaiverse.agent_basics import AgentBasics
from unaiverse.streams.dataprops import StreamType
from unaiverse.uai import (AnswerOutcome, AnswerWithheld, build_form, encode_reply, interpret_reply,
                                parse_message, parts_to_model_text, serialize_block)
from unaiverse.modules.utils import ModuleWrapper, MultiIdentity, HumanModule

FORM = build_form("cena", [{"name": "scelta", "type": "select", "label": "Scelta", "required": True,
                            "options": [{"value": "pizza", "label": "Pizza"},
                                        {"value": "sushi", "label": "Sushi"}]}], form_id="poll1", lang="it")
MESSAGE = "## Pizza o sushi?\n\n" + serialize_block(FORM)
BIG = build_form("prenota", [{"name": "nome", "type": "text", "label": "Nome", "required": True},
                             {"name": "persone", "type": "integer", "label": "Quante persone",
                              "required": True, "min": 1, "max": 10}], form_id="p1", lang="it")
BIG_MESSAGE = "## Prenotazione\n\n" + serialize_block(BIG)


class Spy(torch.nn.Module):
    """A module that records everything it was handed and answers with its fixed texts, one per call."""

    def __init__(self, *answers: str) -> None:
        super().__init__()
        self.answers = list(answers) if answers else ["scelta: Sushi"]
        self.calls = []

    @property
    def seen(self):
        return self.calls[-1] if self.calls else None

    def forward(self, msg: str) -> str:
        self.calls.append(msg)
        return self.answers[min(len(self.calls), len(self.answers)) - 1]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.

    def get_time(self) -> float:
        return self.now


class FakeInteraction:
    def __init__(self, requester, system: bool, created_at: float = 1000., timeout: float = -1.) -> None:
        self.requester = requester
        self.timestamp_created = created_at
        self.timeout = timeout
        self.__system = system

    def is_system(self) -> bool:
        return self.__system


class FakeBehav:
    def __init__(self, wildcards: dict) -> None:
        self.__wildcards = wildcards

    def get_wildcards(self) -> dict:
        return self.__wildcards


class FakeAgent:
    """The smallest agent the gate needs: the real policy methods, and one peer to talk to."""
    UAI_MAX_RETRIES = AgentBasics.UAI_MAX_RETRIES
    UAI_DEADLINE_MARGIN = AgentBasics.UAI_DEADLINE_MARGIN
    uai_preprocess = AgentBasics.uai_preprocess
    uai_postprocess = AgentBasics.uai_postprocess
    uai_answer_fell_short = AgentBasics.uai_answer_fell_short
    uai_time_left = AgentBasics.uai_time_left
    uai_module_echoes = AgentBasics.uai_module_echoes
    uai_merge_reads = AgentBasics.uai_merge_reads
    uai_remember_form = AgentBasics.uai_remember_form
    uai_pending_form = AgentBasics.uai_pending_form
    uai_pending_view = AgentBasics.uai_pending_view
    uai_forget_form = AgentBasics.uai_forget_form

    def __init__(self, human: bool = False, interaction=None, browser: bool = False) -> None:
        self.uai_inbox = {}
        self.uai_writing_to = None
        self.peer = "peer1"
        self.clock = FakeClock()
        self.proc = None  # Set by wrapper()
        self.human = human
        self.browser = browser
        self.interaction = interaction

    def uai_peer(self):
        return self.peer

    def get_current_interaction(self):
        return self.interaction

    def uai_can_reprompt_person(self):  # The real one looks at the platform: here a flag decides
        return self.human and not self.browser


@pytest.fixture(autouse=True)
def a_flag_says_who_is_human(monkeypatch):
    # The policy asks the real helper, which looks at the processor module: here a flag on the fake agent
    # decides, for the duration of each test only
    monkeypatch.setattr(agent_basics_module, "has_human_processor", lambda agent: getattr(agent, "human", False))


class BrokenAgent(FakeAgent):
    def uai_preprocess(self, text):
        raise RuntimeError("boom")

    def uai_postprocess(self, text, spec, **kwargs):
        raise RuntimeError("boom")


class PeerAgent:
    """Exercises the real peer resolution against a controlled interaction and behavior."""
    uai_peer = AgentBasics.uai_peer

    def __init__(self, interaction=None, partner=None, writing_to=None) -> None:
        self.__interaction = interaction
        self.uai_writing_to = writing_to
        self.behav = FakeBehav({})
        self.behav_lone_wolf = FakeBehav({"<partner>": partner} if partner else {})

    def get_current_interaction(self):
        return self.__interaction

    @staticmethod
    def behaving_in_world() -> bool:
        return False


def wrapper(module, agent=None) -> ModuleWrapper:
    agent = agent if agent is not None else FakeAgent()
    w = ModuleWrapper(module=module,
                      proc_inputs=[StreamType(data_type="text")],
                      proc_outputs=[StreamType(data_type="text")],
                      agent=agent)
    agent.proc = w
    return w


def test_the_module_reads_the_alternative_text_and_never_the_json():
    spy = Spy()
    wrapper(spy)(MESSAGE)
    assert "```uai" not in spy.seen and '"type":"form"' not in spy.seen
    assert "Rispondi con righe 'campo: valore'" in spy.seen
    assert "## Pizza o sushi?" in spy.seen


def test_a_clean_answer_leaves_as_one_block_carrying_its_own_projection():
    spy = Spy("scelta: Sushi")
    out = wrapper(spy)(MESSAGE)[0]
    parts = parse_message(out)
    assert len(spy.calls) == 1
    assert out.count("```uai") == 1
    assert len(parts) == 1 and parts[0]["type"] == "reply"
    assert parts[0]["spec"]["to"] == "poll1"
    assert parts[0]["spec"]["values"] == {"scelta": "sushi"}

    # The human rendering of a reply is its alt (contract, section 3.6): the labels of the form, not the
    # words the answer happened to be written with
    assert parts[0]["alt"] == "Scelta: Sushi"


def test_a_model_that_falls_short_is_asked_again_with_what_went_wrong():
    # A required field nobody said anything about, then a value the form does not accept, then a clean one
    spy = Spy("Nome: Mario", "Nome: Mario\nQuante persone: 25", "Nome: Mario\nQuante persone: 4")
    out = wrapper(spy)(BIG_MESSAGE)[0]
    assert len(spy.calls) == 3

    # Each corrective prompt carries the instruction, the previous answer, and what was wrong with it
    assert "Rispondi con righe 'campo: valore'" in spy.calls[1]
    assert "Nome: Mario" in spy.calls[1] and "Quante persone" in spy.calls[1]
    assert "above_max" in spy.calls[2]
    assert "```" not in spy.calls[1] and "```" not in spy.calls[2]

    parts = parse_message(out)
    assert len(parts) == 1 and parts[0]["spec"]["values"] == {"nome": "Mario", "persone": 4}


def test_a_model_answering_in_prose_to_a_form_it_was_just_asked_is_asked_again():
    spy = Spy("Buongiorno! Volentieri.", "scelta: Pizza")
    out = wrapper(spy)(MESSAGE)[0]
    assert len(spy.calls) == 2
    assert parse_message(out)[0]["spec"]["values"] == {"scelta": "pizza"}


def test_when_the_retries_are_spent_one_block_travels_with_values_and_words():
    spy = Spy("Nome: Anna\nQuante persone: 30")
    out = wrapper(spy)(BIG_MESSAGE)[0]
    assert len(spy.calls) == 1 + AgentBasics.UAI_MAX_RETRIES

    # One reply block: the values that could be read, and the words as written in its raw. Whoever asked
    # gets the name, learns the rest is missing, and the value that violates the form is nowhere canonical
    parts = parse_message(out)
    assert len(parts) == 1 and parts[0]["type"] == "reply"
    assert parts[0]["spec"]["values"] == {"nome": "Anna"}
    assert parts[0]["spec"]["raw"] == ["Nome: Anna\nQuante persone: 30"]
    assert "30" not in parts[0]["alt"]

    # And a model reading the answer still sees both faces: the canonical value and the phrasing,
    # including what the values missed
    view = parts_to_model_text(parts)
    assert "nome: Anna" in view and "Quante persone: 30" in view


def test_when_the_retries_are_spent_and_nothing_could_be_read_the_words_travel_inside_the_block():
    agent = FakeAgent()
    agent.uai_remember_form("peer1", MESSAGE)
    assert agent.uai_pending_form() is not None
    spy = Spy("Boh, non saprei")
    out = wrapper(spy, agent=agent)(MESSAGE)[0]
    assert len(spy.calls) == 1 + AgentBasics.UAI_MAX_RETRIES

    # Still one reply block, empty-handed but carrying the words: the answer is over (the remembered
    # form is forgotten), and whoever asked can read how it was phrased
    parts = parse_message(out)
    assert len(parts) == 1 and parts[0]["type"] == "reply"
    assert parts[0]["spec"]["values"] == {} and parts[0]["spec"]["raw"] == ["Boh, non saprei"]
    assert agent.uai_pending_form() is None


def test_a_blank_answer_to_a_form_just_asked_is_a_refusal_and_never_travels():
    # Silence is an answer in ordinary chat, but not to a question just asked: a model that answers
    # nothing is asked again, and one that insists on silence is withheld, never shipped as an empty reply
    spy = Spy("")
    with pytest.raises(AnswerWithheld):
        wrapper(spy)(MESSAGE)
    assert len(spy.calls) == 1 + AgentBasics.UAI_MAX_RETRIES
    assert "Pizza o sushi?" in spy.calls[-1] and "Non soddisfa" in spy.calls[-1]


def test_a_blank_line_with_a_form_merely_pending_is_ordinary_silence():
    # The form arrived earlier and this turn is not about it: an empty answer is the module staying
    # silent, it travels as such and the form keeps waiting
    agent = FakeAgent()
    agent.uai_remember_form("peer1", MESSAGE)
    spy = Spy("")
    assert wrapper(spy, agent=agent)("come va?")[0] == ""
    assert len(spy.calls) == 1
    assert agent.uai_pending_form()["id"] == "poll1"


def test_a_blank_regeneration_never_replaces_the_words_of_an_earlier_attempt():
    # The first answer falls short and the retry gives up entirely: what travels is the best answer the
    # module ever gave, carried as the raw of the failure block, never the blank that came after it
    spy = Spy("scelta: Lasagne", "")
    out = wrapper(spy)(MESSAGE)[0]
    assert len(spy.calls) == 1 + AgentBasics.UAI_MAX_RETRIES
    parts = parse_message(out)
    assert len(parts) == 1 and parts[0]["type"] == "reply"
    assert parts[0]["spec"]["values"] == {} and parts[0]["spec"]["raw"] == ["scelta: Lasagne"]


def test_a_clean_answer_carries_its_words_as_the_raw_of_the_block():
    spy = Spy("scelta: Sushi")
    out = wrapper(spy)(MESSAGE)[0]
    part = parse_message(out)[0]
    assert part["type"] == "reply" and part["spec"]["values"] == {"scelta": "sushi"}
    assert part["spec"]["raw"] == ["scelta: Sushi"]

    # And the raw is what an interpreting receiver reads as the event's words
    event = interpret_reply(out, pending=[FORM])
    assert event.via == "block" and event.raw == "scelta: Sushi"


def test_the_raw_field_is_transparent_and_optional():
    # A block encoded without raw has no such field, byte-identical to the shape the vectors pin
    plain = encode_reply(FORM, {"scelta": "pizza"})
    assert '"raw"' not in plain
    assert "raw" not in parse_message(plain)[0]["spec"]

    # A raw-only failure block still reads as the words wherever a person or a model would see it
    failed = encode_reply(FORM, {}, raw="Boh, non saprei")
    assert parts_to_model_text(parse_message(failed)) == "Boh, non saprei"


def test_a_giant_raw_is_cut_before_it_can_cost_the_values():
    # The words must never cost the values: a raw that would push the block past the fence cap every
    # receiver enforces is cut to fit, so the block survives the gate on arrival and the values are
    # read whole (an uncut raw of this size would degrade the whole block to prose)
    out = encode_reply(FORM, {"scelta": "pizza"}, raw="parola " * 4000)
    part = parse_message(out)[0]
    assert part["type"] == "reply" and part["spec"]["values"] == {"scelta": "pizza"}
    assert len(part["spec"]["raw"]) == 1 and 0 < len(part["spec"]["raw"][0]) < len("parola " * 4000)

    # With several texts, the oldest are dropped first: the latest words survive whole
    out = encode_reply(FORM, {"scelta": "pizza"}, raw=["x" * 6000, "y" * 6000])
    assert parse_message(out)[0]["spec"]["raw"] == ["y" * 6000]


def test_an_answer_is_completed_across_the_corrective_rounds():
    # The first attempt fills half the form; the correction asks ONLY for what is missing (restating
    # what already stands), the second attempt supplies just that, and the merged answer travels as one
    # block whose raw lists both contributing texts, oldest first
    spy = Spy("Nome: Anna", "Quante persone: 4")
    out = wrapper(spy)(BIG_MESSAGE)[0]
    assert len(spy.calls) == 2
    assert "SOLO i campi che mancano" in spy.calls[1] and "Nome: Anna" in spy.calls[1]
    part = parse_message(out)[0]
    assert part["type"] == "reply" and part["spec"]["values"] == {"nome": "Anna", "persone": 4}
    assert part["spec"]["raw"] == ["Nome: Anna", "Quante persone: 4"]


def test_a_fresh_complaint_blocks_the_merge_until_it_is_fixed():
    # Attempt 2 retracts a good value with an out-of-range one: the merge must not ship the stale value
    # over the fresh complaint, and the correction names the complaint, never "you said nothing about it"
    spy = Spy("Quante persone: 4", "Nome: Bea\nQuante persone: 30", "Quante persone: 4")
    out = wrapper(spy)(BIG_MESSAGE)[0]
    assert len(spy.calls) == 3
    assert "non sono accettabili" in spy.calls[2]
    assert "non hai detto nulla su Quante persone" not in spy.calls[2]
    assert parse_message(out)[0]["spec"]["values"] == {"nome": "Bea", "persone": 4}


def test_the_raw_lists_only_the_texts_the_values_were_read_from():
    # An attempt that gave nothing readable (an out-of-range value) is not part of the provenance: the
    # raw of the final block lists the contributing texts only
    spy = Spy("Quante persone: 30", "Nome: Bea\nQuante persone: 4")
    out = wrapper(spy)(BIG_MESSAGE)[0]
    assert len(spy.calls) == 2
    part = parse_message(out)[0]
    assert part["spec"]["values"] == {"nome": "Bea", "persone": 4}
    assert part["spec"]["raw"] == ["Nome: Bea\nQuante persone: 4"]


def test_a_retry_on_a_remembered_form_restates_the_message_that_asked():
    # The form arrived in an earlier message: when the answer falls short, the corrective prompt repeats
    # the rendering of THAT message (its question included), not the form's instruction alone
    agent = FakeAgent()
    agent.uai_remember_form("peer1", MESSAGE)
    spy = Spy("scelta: Lasagne", "scelta: Sushi")
    out = wrapper(spy, agent=agent)("dunque...")[0]
    assert len(spy.calls) == 2 and "Pizza o sushi?" in spy.calls[1]
    assert parse_message(out)[-1]["spec"]["values"] == {"scelta": "sushi"}


def test_an_answer_that_never_satisfied_the_form_is_discarded_when_the_interaction_is_expiring():
    # The asking interaction was created long ago: there is no time for another attempt, and an answer that
    # does not honour the form is not sent to somebody who is about to stop waiting
    agent = FakeAgent(interaction=FakeInteraction("peer1", system=False, created_at=0.))
    agent.clock.now = Custom.DEFAULT_INTER_TIMEOUT - 5.
    spy = Spy("scelta: Poke")
    with pytest.raises(AnswerWithheld):
        wrapper(spy, agent=agent)(MESSAGE)
    assert len(spy.calls) == 1

    # With time to spare the model is asked again as usual
    agent = FakeAgent(interaction=FakeInteraction("peer1", system=False, created_at=0.))
    agent.clock.now = 10.
    spy = Spy("scelta: Poke", "scelta: Sushi")
    wrapper(spy, agent=agent)(MESSAGE)
    assert len(spy.calls) == 2


def test_time_left_follows_the_interaction_that_asked():
    agent = FakeAgent()
    assert agent.uai_time_left() == Custom.DEFAULT_INTER_TIMEOUT - AgentBasics.UAI_DEADLINE_MARGIN
    agent = FakeAgent(interaction=FakeInteraction("peer1", system=False, created_at=900., timeout=60.))
    assert agent.uai_time_left() == 60. - 100. - AgentBasics.UAI_DEADLINE_MARGIN
    agent = FakeAgent(interaction=FakeInteraction(None, system=True, created_at=0.))
    assert agent.uai_time_left() == Custom.DEFAULT_INTER_TIMEOUT - AgentBasics.UAI_DEADLINE_MARGIN


def test_a_message_without_blocks_is_untouched():
    spy = Spy("ciao")
    out = wrapper(spy)("Buongiorno, come va?")
    assert spy.seen == "Buongiorno, come va?"
    assert out[0] == "ciao"


def test_an_echoing_module_never_has_its_input_rewritten():
    # This is the answer a person composes in the web application: rewriting it would destroy the block the
    # world is waiting for, and a relay would corrupt every form it forwards
    answer = encode_reply(FORM, {"scelta": "sushi"})
    assert wrapper(MultiIdentity())(answer)[0] == answer
    assert MultiIdentity.UAI_ECHOES_INPUT and HumanModule.UAI_ECHOES_INPUT

    # And a form travelling through a relay stays a form
    assert wrapper(MultiIdentity())(MESSAGE)[0] == MESSAGE

    # A relay forwarding somebody else's words that fall short of a pending form is not asked again: it is
    # not the author, and what it carries travels as it is
    agent = FakeAgent()
    agent.uai_remember_form("peer1", MESSAGE)
    assert wrapper(MultiIdentity(), agent=agent)("scelta: Poke")[0] == "scelta: Poke"


def test_a_person_answers_what_is_on_their_screen():
    # The form was shown to them and never went through their processor: it was remembered when it arrived,
    # and their own words are encoded on the way out, which is what an echoing module makes possible
    agent = FakeAgent(human=True)
    agent.uai_remember_form("peer1", MESSAGE)
    out = wrapper(HumanModule(), agent=agent)("scelta: Sushi")[0]
    parts = parse_message(out)
    assert parts[-1]["spec"]["to"] == "poll1" and parts[-1]["spec"]["values"] == {"scelta": "sushi"}
    assert parts[-1]["alt"] == "Scelta: Sushi"

    # A form is answered once: what they write next is their own business again
    assert wrapper(HumanModule(), agent=agent)("scelta: Pizza")[0] == "scelta: Pizza"


def test_a_person_who_falls_short_is_told_and_nothing_travels_until_they_answer_again(caplog):
    agent = FakeAgent(human=True)
    agent.uai_remember_form("peer1", BIG_MESSAGE)

    # A required field missing: told, withheld, the form stays pending
    with pytest.raises(AnswerWithheld):
        wrapper(HumanModule(), agent=agent)("Nome: Mario")
    assert agent.uai_pending_form()["id"] == "p1"

    # Chatting in the meantime is their own business, the form still waits
    assert wrapper(HumanModule(), agent=agent)("un attimo")[0] == "un attimo"
    assert agent.uai_pending_form()["id"] == "p1"

    # The next proper answer is encoded and the form is over
    out = wrapper(HumanModule(), agent=agent)("Nome: Mario\nQuante persone: 4")[0]
    assert parse_message(out)[-1]["spec"]["values"] == {"nome": "Mario", "persone": 4}
    assert agent.uai_pending_form() is None


def test_an_answer_that_is_already_canonical_is_left_alone_and_closes_the_form():
    # The web application encodes the answer itself, partial or not: appending a second block would be
    # nonsense, and a skip is a legitimate answer there
    agent = FakeAgent(human=True, browser=True)
    agent.uai_remember_form("peer1", BIG_MESSAGE)
    answer = encode_reply(BIG, {"nome": "Mario"})
    assert wrapper(HumanModule(), agent=agent)(answer)[0] == answer

    # The form is over: what is typed next is plain chat, not a second answer
    assert agent.uai_pending_form() is None
    assert wrapper(HumanModule(), agent=agent)("Quante persone: 4")[0] == "Quante persone: 4"

    # A block to some other form (an older card in the timeline) passes too, and the remembered one stays
    agent.uai_remember_form("peer1", BIG_MESSAGE)
    other = encode_reply(FORM, {"scelta": "sushi"})
    assert wrapper(HumanModule(), agent=agent)(other)[0] == other
    assert agent.uai_pending_form()["id"] == "p1"


def test_the_same_short_answer_meets_three_different_fates():
    # A model is asked again
    spy = Spy("Nome: Mario", "Nome: Mario\nQuante persone: 4")
    out = wrapper(spy)(BIG_MESSAGE)[0]
    assert len(spy.calls) == 2 and parse_message(out)[0]["type"] == "reply"

    # A person at a terminal is told, and nothing travels
    agent = FakeAgent(human=True)
    agent.uai_remember_form("peer1", BIG_MESSAGE)
    with pytest.raises(AnswerWithheld):
        wrapper(HumanModule(), agent=agent)("Nome: Mario")

    # A person in the browser cannot be told: their words travel as written, the form stays open for the
    # widget, and a complete answer typed by hand is still encoded
    agent = FakeAgent(human=True, browser=True)
    agent.uai_remember_form("peer1", BIG_MESSAGE)
    assert wrapper(HumanModule(), agent=agent)("Nome: Mario")[0] == "Nome: Mario"
    assert agent.uai_pending_form()["id"] == "p1"
    out = wrapper(HumanModule(), agent=agent)("Nome: Mario\nQuante persone: 4")[0]
    assert parse_message(out)[-1]["spec"]["values"] == {"nome": "Mario", "persone": 4}
    assert agent.uai_pending_form() is None


def test_the_terminal_is_where_a_person_can_be_asked_again(monkeypatch):
    class Real:
        uai_can_reprompt_person = AgentBasics.uai_can_reprompt_person
        human = True
    monkeypatch.setattr(agent_basics_module.sys, "platform", "linux")
    assert Real().uai_can_reprompt_person()
    monkeypatch.setattr(agent_basics_module.sys, "platform", "emscripten")
    assert not Real().uai_can_reprompt_person()


def test_the_inbox_remembers_the_last_form_per_peer_forgets_it_by_id_and_in_time():
    agent = FakeAgent()
    agent.uai_remember_form("peer1", "just prose, no form here")
    assert agent.uai_inbox == {}

    agent.uai_remember_form("peer1", MESSAGE)
    assert agent.uai_pending_form()["id"] == "poll1"

    other = build_form("altro", [{"name": "a", "type": "text"}], form_id="second")
    agent.uai_remember_form("peer1", serialize_block(other))
    assert agent.uai_pending_form()["id"] == "second"

    # Forgetting goes by id, not by the very object that was remembered: a form handed to the module in the
    # same call is parsed anew, and must still clear the one that arrived earlier as a message
    agent.uai_forget_form({"id": "poll1"})
    assert agent.uai_pending_form()["id"] == "second"
    agent.uai_forget_form(dict(other))
    assert agent.uai_pending_form() is None

    # Nobody waits for an answer longer than an interaction can live
    agent.uai_remember_form("peer1", MESSAGE)
    agent.clock.now += Custom.DEFAULT_INTER_TIMEOUT + 1.
    assert agent.uai_pending_form() is None

    for i in range(64):
        agent.uai_remember_form(f"peer{i}", MESSAGE)
    assert len(agent.uai_inbox) <= 32


def test_a_form_asked_in_the_same_call_clears_the_copy_that_arrived_as_a_message():
    # The message was received (remembered) and then handed to the model in the same turn
    agent = FakeAgent()
    agent.uai_remember_form("peer1", MESSAGE)
    wrapper(Spy("scelta: Sushi"), agent=agent)(MESSAGE)
    assert agent.uai_pending_form() is None


def test_the_peer_of_a_turn_is_whoever_asked_or_the_partner():
    # Somebody asked: it is theirs
    assert PeerAgent(interaction=FakeInteraction("asker", system=False)).uai_peer() == "asker"

    # A person at a keyboard writes on their own initiative: the turn is a system one, and the peer is the
    # one the node put their words on the way to
    assert PeerAgent(interaction=FakeInteraction(None, system=True), writing_to="mate").uai_peer() == "mate"

    # Failing that, the partner the behavior is pointing at
    assert PeerAgent(interaction=FakeInteraction(None, system=True), partner="mate").uai_peer() == "mate"
    assert PeerAgent(interaction=None, partner="mate").uai_peer() == "mate"
    assert PeerAgent(interaction=None).uai_peer() is None

    # Somebody asking always wins over both
    assert PeerAgent(interaction=FakeInteraction("asker", system=False),
                     writing_to="mate", partner="other").uai_peer() == "asker"


class PrepareStub:
    """What AgentBasics.prepare_stdin_if_human touches, and nothing else (see tests/test_human_routing.py)."""

    def __init__(self) -> None:
        self.proc_human_peer_id_to_interaction = {}
        self.uai_writing_to = None

    def set_default_stdin_binding(self, public=None):
        pass


def test_the_node_tells_the_agent_who_the_person_is_writing_to():
    # This is the one place that knows it: the interactive loop of the node passes the target peer here, and
    # a person's answer would otherwise have no peer to be matched against, its turn being a system one
    stub = PrepareStub()
    AgentBasics.prepare_stdin_if_human(stub, True, "the-peer")
    assert stub.uai_writing_to == "the-peer"


class LenientAgent(FakeAgent):
    """A world that accepts a judgement on whoever was named: nothing is ever asked again."""

    def uai_answer_fell_short(self, text, spec, event, attempt, model_view=None):
        return AnswerOutcome()


def test_a_world_can_decide_that_falling_short_is_fine():
    spy = Spy("Nome: Mario")
    assert wrapper(spy, agent=LenientAgent())(BIG_MESSAGE)[0] == "Nome: Mario"
    assert len(spy.calls) == 1


class EndlessAgent(FakeAgent):
    """A broken override that asks again forever."""

    def uai_answer_fell_short(self, text, spec, event, attempt, model_view=None, attempts=None):
        return AnswerOutcome(retry="again")


def test_a_runaway_policy_is_stopped_by_the_wrapper():
    spy = Spy("Nome: Mario")
    wrapper(spy, agent=EndlessAgent())(BIG_MESSAGE)
    assert len(spy.calls) == 1 + ModuleWrapper.UAI_HARD_CAP


def test_a_failing_world_override_costs_nothing_but_the_encoding():
    # A world may override either method: a broken one must never cost the step or the answer
    out = wrapper(Spy("scelta: Sushi"), agent=BrokenAgent())(MESSAGE)
    assert out[0] == "scelta: Sushi"


def test_a_wrapper_without_an_agent_is_left_alone():
    spy = Spy()
    ModuleWrapper(module=spy, proc_inputs=[StreamType(data_type="text")],
                  proc_outputs=[StreamType(data_type="text")])(MESSAGE)
    assert spy.seen == MESSAGE
