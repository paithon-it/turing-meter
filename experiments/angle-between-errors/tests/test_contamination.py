"""A11: the permutation moves the correct option to another letter, and the answers map back to the original order."""
from angle.contamination import back, rotate, swap, target

ITEM = {"id": "x-1", "task": "chatbench", "options": ["a", "b", "c", "d"], "correct_option": 2}


def test_target_is_another_position_and_fixed():
    for i in range(50):
        t = target(f"id-{i}", 4, 1)
        assert t != 1 and 0 <= t < 4
        assert t == target(f"id-{i}", 4, 1)


def test_rotation_keeps_the_options_and_moves_the_key():
    r = rotate(ITEM)
    assert sorted(r["options"]) == ITEM["options"]
    assert r["options"][r["correct_option"]] == "c"
    assert r["correct_option"] != ITEM["correct_option"]
    assert [ITEM["options"][i] for i in r["original"]] == r["options"]


def test_answers_map_back_to_the_original_order():
    r = rotate(ITEM)
    prob = [0.0] * 4
    prob[r["options"].index("b")] = 1.0          # the machine chose option "b", wherever it was shown
    row = back({"id": "x-1", "prob": prob, "correct_option": r["correct_option"]}, r)
    assert row["prob"] == [0.0, 1.0, 0.0, 0.0]
    assert row["correct_option"] == 2


def test_swap_alphanli():
    s = swap({"id": "s", "task": "alphanli", "options": ["p", "q"], "correct_option": 0})
    assert s["options"] == ["q", "p"] and s["correct_option"] == 1
    row = back({"id": "s", "prob": [0.9, 0.1], "correct_option": 1}, s)
    assert row["prob"] == [0.1, 0.9] and row["correct_option"] == 0
