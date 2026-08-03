"""Decoder-side behaviour that does not need a GPU.

`hotwords_note` is the interesting one: hotwords are the only context that reaches
every 30-second window of a sung file, so they are what makes a lyric decode
coherent — and faster-whisper truncates them without telling anyone.
"""

from sgen.asr import Recognizer
from sgen.config import AsrConfig


class FakeTokenizer:
    """One token per character, which is enough to count with."""

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=True):
        return self.Encoding(list(range(len(text))))


class FakeModel:
    max_length = 448  # what Whisper actually uses

    def __init__(self):
        self.hf_tokenizer = FakeTokenizer()


def recognizer(hotwords):
    """A Recognizer without the model load — nothing here touches CUDA."""
    rec = Recognizer.__new__(Recognizer)
    rec.cfg = AsrConfig(hotwords=hotwords)
    rec._model = FakeModel()
    return rec


BUDGET = FakeModel.max_length // 2 - 1  # 223, matching faster-whisper's get_prompt


def test_no_note_without_hotwords():
    assert recognizer(None).hotwords_note() is None
    assert recognizer("").hotwords_note() is None
    assert recognizer("   ").hotwords_note() is None


def test_no_note_for_a_list_that_fits():
    assert recognizer("Thomas, Oaxaca, Kreuzberg").hotwords_note() is None
    assert recognizer("x" * (BUDGET - 1)).hotwords_note() is None


def test_a_pasted_lyric_sheet_says_what_was_dropped():
    """Silence here meant the tail of a long paste looked like it was in effect."""
    note = recognizer("x" * (BUDGET * 4)).hotwords_note()
    assert note is not None
    assert str(BUDGET) in note, note
    assert "%" in note, note
    # Actionable, not just a complaint.
    assert "first" in note.lower(), note


def test_the_note_grows_with_the_overflow():
    """A list twice the budget should report a bigger loss than one just over."""
    def dropped(length):
        note = recognizer("x" * length).hotwords_note()
        return int(note.split("about ")[1].split("%")[0])

    assert dropped(BUDGET * 4) > dropped(BUDGET + 40)


def test_the_budget_comes_from_the_model_not_a_constant():
    """If the model's context changes, the warning has to move with it."""
    rec = recognizer("x" * 300)
    rec._model.max_length = 4096
    assert rec.hotwords_note() is None


def test_no_note_when_the_model_is_released():
    """close() drops the model; a note must not resurrect it."""
    rec = recognizer("x" * 5000)
    rec._model = None
    assert rec.hotwords_note() is None
