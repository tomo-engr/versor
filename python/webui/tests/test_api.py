# WebUI API tests. These drive the real FastAPI app (which spawns the
# real versor-cli serve engine), so they need a release build of
# versor-cli and the weights the app defaults to. Key checks:
#   - each endpoint responds correctly
#   - analyze via the WebUI equals versor-cli invoked directly on the
#     same position (engine agreement)
#   - hint values are sign/scale consistent with playing the move and
#     analyzing the child (the side-to-move sign bug guard)

import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
WEBUI = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WEBUI))
BIN = os.path.join(REPO, "target", "release", "versor-cli")
# Same default as python/webui/engine.py: the current best weights, so
# the engine-agreement test compares like with like.
WEIGHTS = os.environ.get("VERSOR_WEIGHTS", os.path.join(REPO, "models", "pattern_best.bin"))

sys.path.insert(0, WEBUI)

INITIAL = "---------------------------OX------XO---------------------------"


@pytest.fixture(scope="module")
def client():
    if not os.path.exists(BIN):
        pytest.skip("versor-cli release binary not built")
    import app as appmod

    with TestClient(appmod.app) as c:
        yield c
    appmod.engine.close()


def direct_analyze(board, stm, depth):
    # Run versor-cli serve directly for one analyze request.
    req = json.dumps({"cmd": "analyze", "board": board, "stm": stm, "depth": depth})
    proc = subprocess.run(
        [BIN, "serve", "--weights", WEIGHTS, "--tt", "64"],
        input=req + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    # First line is the readiness banner, second is the response.
    return json.loads(lines[1])


def test_initial(client):
    r = client.get("api/initial").json()
    assert r["ok"] is True
    assert sorted(r["legal"]) == ["c4", "d3", "e6", "f5"]
    assert r["black"] == 2 and r["white"] == 2


def test_play_legal_and_illegal(client):
    ok = client.post("api/play", json={"board": INITIAL, "stm": "X", "move": "d3"}).json()
    assert ok["stm"] == "O"
    assert ok["black"] == 4 and ok["white"] == 1
    bad = client.post("api/play", json={"board": INITIAL, "stm": "X", "move": "a1"})
    assert bad.status_code == 400


def test_analyze_matches_direct_engine(client):
    # WebUI analyze must equal versor-cli invoked directly (same board,
    # same depth) in best move and value.
    for moves, depth in [("", 8), ("f5d6c3", 8), ("f5f6e6f4", 6)]:
        board, stm = INITIAL, "X"
        for i in range(0, len(moves), 2):
            res = client.post(
                "api/play", json={"board": board, "stm": stm, "move": moves[i : i + 2]}
            ).json()
            board, stm = res["board"], res["stm"]
        web = client.post(
            "api/analyze", json={"board": board, "stm": stm, "depth": depth}
        ).json()
        direct = direct_analyze(board, stm, depth)
        assert web["value"] == direct["value"], (moves, web, direct)
        assert web["best_move"] == direct["best_move"], (moves, web, direct)


def test_hints_sign_consistent(client):
    # For each legal move, the hint value (mover's perspective) must
    # equal the negation of analyzing the child position at one ply
    # shallower. This guards the side-to-move sign convention.
    depth = 6
    board, stm = INITIAL, "X"
    hints = client.post(
        "api/hints", json={"board": board, "stm": stm, "depth": depth}
    ).json()["hints"]
    assert len(hints) == 4
    for item in hints:
        played = client.post(
            "api/play", json={"board": board, "stm": stm, "move": item["move"]}
        ).json()
        child = client.post(
            "api/analyze",
            json={"board": played["board"], "stm": played["stm"], "depth": depth - 1},
        ).json()
        assert item["value"] == -child["value"], item


def test_hints_max_equals_analyze(client):
    depth = 6
    hints = client.post(
        "api/hints", json={"board": INITIAL, "stm": "X", "depth": depth}
    ).json()["hints"]
    a = client.post(
        "api/analyze", json={"board": INITIAL, "stm": "X", "depth": depth}
    ).json()
    assert max(h["value"] for h in hints) == a["value"]


def test_import_and_evalgraph(client):
    moves = "f5d6c3d3c4"
    imp = client.post("api/import", json={"moves": moves}).json()
    assert len(imp["states"]) == 5
    assert imp["final"]["board"] == imp["states"][-1]["board"]
    graph = client.post("api/evalgraph", json={"moves": moves, "depth": 4}).json()
    # ply 0 (initial) plus one point per move.
    assert len(graph["points"]) == 6
    assert graph["points"][0]["ply"] == 0


def test_missing_weights_reported():
    # Without a weight file the engine cannot start; the error says why
    # (the engine's own message names the file) instead of only failing.
    if not os.path.exists(BIN):
        pytest.skip("versor-cli release binary not built")
    import engine as engmod

    missing = os.path.join(REPO, "models", "no_such_weights.bin")
    eng = engmod.Engine(binary=BIN, weights=missing, mpc="")
    with pytest.raises(engmod.EngineError) as e:
        eng.request({"cmd": "initial"})
    assert "did not report ready" in str(e.value)
    assert "no_such_weights.bin" in str(e.value)
