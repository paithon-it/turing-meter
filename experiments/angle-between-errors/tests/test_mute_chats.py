"""The chats without an answer (PROTOCOL.md, «After the seventh review», B): matching and intervals, before any run."""
import math

from angle import mute_chats as mc


def test_an_option_is_found_as_a_whole_token():
    opts = ["1/64", "15/64", "1/4", "3/8"]
    assert mc.named_options("So the probability is 15/64.", opts) == [1]
    assert mc.named_options("The answer is \\(1/4\\)", opts) == [2]
    assert mc.named_options("11/48 is not it", opts) == []           # 1/4 inside 11/48 is not a match
    assert mc.named_options("Options: 1/64, 15/64, 1/4 or 3/8", opts) == [0, 1, 2, 3]


def test_words_and_case():
    opts = ["Frequency", "amplitude", "wavelength", "speed"]
    assert mc.named_options("Pitch is related to its FREQUENCY.", opts) == [0]
    assert mc.named_options("frequencies differ", opts) == []          # not the whole word


def test_short_options_are_skipped():
    assert mc.named_options("x = 2 and y = 3", ["2", "3", "4", "5"]) == []


def test_machine_text_ignores_the_person():
    h = [{"role": "User", "content": "is it frequency?"}, {"role": "AI", "content": "  "}, {"role": "AI", "content": "Yes."}]
    assert mc.machine_text(h) == "Yes."
    assert mc.machine_text([{"role": "User", "content": "hello"}]) == ""


def test_wilson_matches_the_closed_form():
    lo, hi = mc.wilson(10, 50)
    assert math.isclose(lo, 0.1124, abs_tol=1e-3) and math.isclose(hi, 0.3304, abs_tol=1e-3)
    assert abs(mc.wilson(0, 50)[0]) < 1e-12
