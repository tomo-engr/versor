# versor WebUI backend (FastAPI). Serves the play/analysis frontend and
# proxies engine requests to a persistent `versor-cli serve` process.
# Can run behind a reverse proxy under a path prefix (uvicorn
# --root-path /<prefix>); all frontend fetches are relative so the
# prefix is transparent.
#
# The engine (versor-core) is the sole authority for rules and scoring:
# every legal-move set, move application and evaluation comes from it,
# so the frontend never re-implements othello logic.

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import Engine, EngineError

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

engine = Engine()


@asynccontextmanager
async def lifespan(_app):
    yield
    engine.close()


app = FastAPI(title="versor WebUI", lifespan=lifespan)

INITIAL_BOARD = "---------------------------OX------XO---------------------------"


class PositionReq(BaseModel):
    board: str
    stm: str


class PlayReq(BaseModel):
    board: str
    stm: str
    move: str


class AnalyzeReq(BaseModel):
    board: str
    stm: str
    depth: int | None = None
    nodes: int | None = None
    ms: int | None = None


class HintsReq(BaseModel):
    board: str
    stm: str
    depth: int | None = None


class ImportReq(BaseModel):
    moves: str  # f5d6-style concatenated 2-char moves


class EvalGraphReq(BaseModel):
    moves: str
    depth: int = 4


def _call(obj):
    try:
        resp = engine.request(obj)
    except EngineError as e:
        raise HTTPException(status_code=500, detail=f"engine error: {e}")
    if not resp.get("ok", False):
        raise HTTPException(status_code=400, detail=resp.get("error", "engine rejected request"))
    return resp


@app.get("/api/initial")
def initial():
    return _call({"cmd": "state", "board": INITIAL_BOARD, "stm": "X"})


@app.post("/api/state")
def state(req: PositionReq):
    return _call({"cmd": "state", "board": req.board, "stm": req.stm})


@app.post("/api/play")
def play(req: PlayReq):
    return _call({"cmd": "play", "board": req.board, "stm": req.stm, "move": req.move})


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    obj = {"cmd": "analyze", "board": req.board, "stm": req.stm}
    if req.nodes is not None:
        obj["nodes"] = req.nodes
    elif req.ms is not None:
        obj["ms"] = req.ms
    else:
        obj["depth"] = req.depth if req.depth is not None else 8
    return _call(obj)


@app.post("/api/hints")
def hints(req: HintsReq):
    obj = {"cmd": "hints", "board": req.board, "stm": req.stm}
    if req.depth is not None:
        obj["depth"] = req.depth
    return _call(obj)


# Replay a move string from the initial position, inserting passes
# automatically. Returns the sequence of states (one per applied move)
# and the final position, so the frontend can rebuild history/graph.
def _replay(moves_str):
    seq = moves_str.strip().lower().replace(" ", "").replace("\n", "")
    if len(seq) % 2 != 0:
        raise HTTPException(status_code=400, detail="odd number of coordinate characters")
    board, stm = INITIAL_BOARD, "X"
    states = []
    for i in range(0, len(seq), 2):
        mv = seq[i : i + 2]
        # Auto-pass if the side to move has no legal move.
        st = _call({"cmd": "state", "board": board, "stm": stm})
        if st["must_pass"]:
            passed = _call({"cmd": "play", "board": board, "stm": stm, "move": "pass"})
            board, stm = passed["board"], passed["stm"]
        res = _call({"cmd": "play", "board": board, "stm": stm, "move": mv})
        board, stm = res["board"], res["stm"]
        states.append({"move": mv, **res})
    return states


@app.post("/api/import")
def import_game(req: ImportReq):
    states = _replay(req.moves)
    final = states[-1] if states else _call(
        {"cmd": "state", "board": INITIAL_BOARD, "stm": "X"}
    )
    return {"states": states, "final": final}


@app.post("/api/evalgraph")
def evalgraph(req: EvalGraphReq):
    states = _replay(req.moves)
    depth = max(1, min(req.depth, 12))
    points = []
    # Evaluate the initial position and each resulting position; report
    # the value from black's perspective so the graph has a fixed axis.
    board, stm = INITIAL_BOARD, "X"
    a = _call({"cmd": "analyze", "board": board, "stm": stm, "depth": depth})
    points.append({"ply": 0, "value_black": a["value"] if stm == "X" else -a["value"]})
    for i, s in enumerate(states):
        board, stm = s["board"], s["stm"]
        if s["game_over"]:
            # Use the exact disc difference at terminal nodes.
            vb = s["black"] - s["white"]
            # Award empties to the winner to match final_score scale.
            empty = s["empty"]
            if s["black"] > s["white"]:
                vb += empty
            elif s["white"] > s["black"]:
                vb -= empty
            points.append({"ply": i + 1, "value_black": vb})
            continue
        a = _call({"cmd": "analyze", "board": board, "stm": stm, "depth": depth})
        vb = a["value"] if stm == "X" else -a["value"]
        points.append({"ply": i + 1, "value_black": vb})
    return {"points": points}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
