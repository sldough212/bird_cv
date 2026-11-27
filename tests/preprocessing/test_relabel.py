from typing import Any

from bird_cv.preprocessing.relabel import activity_groups


def test_activity_groups_splits_head_move_by_span_length() -> None:
    head_move = {
        "id": "h",
        "type": "timelinelabels",
        "value": {
            # 15 frames, 16 frames, 1 frame
            "ranges": [
                {"start": 1, "end": 15},
                {"start": 20, "end": 35},
                {"start": 40, "end": 40},
            ],
            "timelinelabels": ["head_move"],
        },
    }
    preening = {
        "id": "p",
        "type": "timelinelabels",
        "value": {"ranges": [{"start": 50, "end": 52}], "timelinelabels": ["preening"]},
    }
    tasks: list[dict[str, Any]] = [{"annotations": [{"result": [head_move, preening]}]}]

    result = activity_groups(tasks)[0]["annotations"][0]["result"]

    assert [(r["value"]["timelinelabels"], r["value"]["ranges"]) for r in result] == [
        (["restful"], [{"start": 1, "end": 15}, {"start": 40, "end": 40}]),
        (["awake"], [{"start": 20, "end": 35}]),
        (["awake"], [{"start": 50, "end": 52}]),
    ]
    assert [r["id"] for r in result] == ["h", "h_1", "p"]
    # The export itself is untouched
    assert tasks[0]["annotations"][0]["result"][0]["value"]["timelinelabels"] == [
        "head_move"
    ]
