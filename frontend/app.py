"""Estimate screen: give a game (paste / file / play on a board) -> estimated ratings.

Run:  streamlit run app.py        (API address: API_URL env var)

The two chessboards (click-to-move input and the replay viewer) are small custom
Streamlit components in components/board and components/replay (plain HTML/JS,
no build step). Keep that folder next to this file.
"""
import math
import time
from pathlib import Path

import altair as alt
import chess
import chess.pgn
import chess.svg
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from api_client import ApiError, friendly, get_config, predict, ready, split_games, validate

st.set_page_config(page_title="Chess Rating Estimator", page_icon="♟️", layout="wide")

_COMP_DIR = Path(__file__).parent / "components"
chess_board = components.declare_component("chess_board", path=str(_COMP_DIR / "board"))
game_replay = components.declare_component("game_replay", path=str(_COMP_DIR / "replay"))

SAMPLE = ('[White "Example White"]\n[Black "Example Black"]\n[WhiteElo "1500"]\n'
          '[BlackElo "1480"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 '
          '5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0')
RESULTS = {"White won": "1-0", "Black won": "0-1", "Draw": "1/2-1/2", "Not known": "*"}
MARKS = {"inaccuracy": ("?!", "blue"), "mistake": ("?", "orange"), "blunder": ("??", "red")}


@st.cache_data(ttl=60, show_spinner=False)
def cached_config():
    return get_config()


@st.cache_data(show_spinner=False)
def cached_validate(pgn):
    return validate(pgn)


@st.cache_data(show_spinner=False)
def cached_split(text):
    return split_games(text)


@st.cache_data(show_spinner=False)
def piece_svgs():
    """{'P': <svg>, 'n': <svg>, ...} so the browser components can draw the pieces."""
    return {chess.Piece(t, c).symbol(): chess.svg.piece(chess.Piece(t, c))
            for c in (chess.WHITE, chess.BLACK) for t in range(1, 7)}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Service status")
    status = ready()
    if status.get("status") == "ready":
        st.success("Ready")
    elif status.get("status") == "unreachable":
        st.error("The server can't be reached.")
    else:
        st.warning("The service is not fully ready.")
        st.caption(f"Chess engine: {'ok' if status.get('engine') else 'down'} · "
                   f"Rating model: {'ok' if status.get('model') else 'down'}")
    st.button("Refresh")

st.title("♟️ Chess Rating Estimator")
st.write("Give us one chess game. We analyse every move with a chess engine and estimate "
         "how strong each player is.")

try:
    cfg = cached_config()
except ApiError as err:
    st.error(friendly(err))
    st.stop()


# ------------------------------------------------------------------- inputs
def show_check(pgn):
    """Validate with the API; return the PGN if usable, else None."""
    try:
        r = cached_validate(pgn)
    except ApiError as err:
        st.error(friendly(err))
        return None
    if not r["valid"]:
        st.error(r["error"]["message"])
        return None
    result = r["result"] if r["result_known"] else "unknown"
    st.success(f"Valid game: {r['plies']} half-moves, result {result}.")
    for w in r["warnings"]:
        st.info(w)
    return pgn


def paste_input():
    def insert_sample():
        st.session_state.paste_text = SAMPLE

    text = st.text_area("Paste one game in PGN format", key="paste_text", height=220)
    st.button("Insert example game", on_click=insert_sample)
    if not text.strip():
        return None
    if len(text.encode("utf-8")) > cfg["max_pgn_bytes"]:
        st.error(f"That text is too long (limit {cfg['max_pgn_bytes']:,} bytes).")
        return None
    return show_check(text)


def upload_input():
    file = st.file_uploader("Choose a .pgn or .txt file", type=["pgn", "txt"])
    if file is None:
        return None
    if file.size > cfg["max_upload_bytes"]:
        st.error(f"That file is too large (limit {cfg['max_upload_bytes'] // 1_000_000} MB).")
        return None
    try:
        found = cached_split(file.getvalue().decode("utf-8", errors="ignore"))
    except ApiError as err:
        st.error(friendly(err))
        return None
    games = found["games"]
    if found["truncated"]:
        st.warning(f"Only the first {found['limit']} games of the file are listed.")

    def label(g):
        tag = "" if g["usable"] else " — can't be used"
        return (f"#{g['index']}  {g['white']} vs {g['black']}, {g['result']}, "
                f"{g['plies']} half-moves{tag}")

    chosen = games[0] if len(games) == 1 else st.selectbox(
        f"This file has {len(games)} games. Pick one:", games, format_func=label)
    if not chosen["usable"]:
        st.error(chosen["problem"])
        return None
    st.success(f"Selected: {label(chosen)}")
    return chosen["pgn"]


def board_input():
    ss = st.session_state
    ss.setdefault("moves", [])
    ss.setdefault("flip", False)
    ss.setdefault("confirm_reset", False)
    ss.setdefault("last_click", None)   # nonce of the last board click we processed

    def position():
        b = chess.Board()
        for u in ss.moves:
            b.push_uci(u)
        return b

    def undo():
        if ss.moves:
            ss.moves.pop()

    def reset():
        ss.moves, ss.confirm_reset = [], False

    board = position()
    left, right = st.columns([1, 1])
    with left:
        click = chess_board(
            fen=board.board_fen(),
            turn="w" if board.turn == chess.WHITE else "b",
            legal=[m.uci() for m in board.legal_moves],
            flipped=ss.flip,
            last=board.peek().uci() if board.move_stack else None,
            check=chess.square_name(board.king(board.turn)) if board.is_check() else None,
            pieces=piece_svgs(),
            key="board", default=None)
        # The component keeps returning its last value on every rerun; the nonce makes
        # sure each click is applied once. The move is re-checked here, never trusted.
        if click and click.get("nonce") != ss.last_click:
            ss.last_click = click["nonce"]
            try:
                move = chess.Move.from_uci(click["uci"])
            except (ValueError, KeyError, TypeError):
                move = None
            if move is not None and move in board.legal_moves:
                ss.moves.append(move.uci())
                st.rerun()
    with right:
        side = "White" if board.turn == chess.WHITE else "Black"
        if board.legal_moves:
            st.markdown(f"**{side} to move.** Click a piece, then click the square you want "
                        "to move it to.")
        else:
            st.info(f"The game is over ({board.result()}).")
        c1, c2, c3 = st.columns(3)
        c1.button("Undo", on_click=undo, disabled=not ss.moves)
        if c2.button("Reset", disabled=not ss.moves):
            ss.confirm_reset = True
        ss.flip = c3.toggle("Flip board", value=ss.flip)
        if ss.confirm_reset:
            st.warning("Clear all moves?")
            y, n = st.columns(2)
            y.button("Yes, clear", on_click=reset)
            if n.button("Keep them"):
                ss.confirm_reset = False
                st.rerun()
        need = cfg["min_plies"]
        st.progress(min(len(ss.moves) / need, 1.0),
                    text=f"{len(ss.moves)} half-moves played — at least {need} needed "
                         f"({math.ceil(need / 2)} full moves)")
        result = st.selectbox("How did the game end?", list(RESULTS))
        names = st.columns(2)
        white = names[0].text_input("White's name (optional)")
        black = names[1].text_input("Black's name (optional)")
        if ss.moves:
            sans, tmp = [], chess.Board()
            for u in ss.moves:
                m = chess.Move.from_uci(u)
                sans.append(tmp.san(m))
                tmp.push(m)
            st.caption("Moves: " + " ".join(
                (f"{i // 2 + 1}. " if i % 2 == 0 else "") + s for i, s in enumerate(sans)))
    if len(ss.moves) < cfg["min_plies"]:
        return None
    game = chess.pgn.Game()
    game.headers["White"], game.headers["Black"] = white or "White", black or "Black"
    game.headers["Result"] = RESULTS[result]
    node = game
    for u in ss.moves:
        node = node.add_variation(chess.Move.from_uci(u))
    return str(game)


st.subheader("1. Give us a game")
mode = st.radio("How do you want to enter it?",
                ["Paste PGN", "Upload a file", "Play it on a board"], horizontal=True)
pgn = {"Paste PGN": paste_input, "Upload a file": upload_input,
       "Play it on a board": board_input}[mode]()

# ------------------------------------------------------------------ analyse
st.subheader("2. Analyse")
player = st.radio("Estimate the rating of", ["both", "white", "black"], horizontal=True,
                  format_func=lambda p: {"both": "Both players", "white": "White only",
                                         "black": "Black only"}[p])
if st.button("Analyse game", type="primary", disabled=pgn is None):
    st.session_state.pop("result", None)
    try:
        for attempt in range(1, 4):
            with st.spinner("Analysing every move… long games can take up to a minute."):
                try:
                    st.session_state.result = predict(pgn, player)
                    break
                except ApiError as err:
                    if err.code != "engine_busy" or attempt == 3:
                        raise
            st.info("The service is busy, trying again in a few seconds…")
            time.sleep(5)
    except ApiError as err:
        st.error(friendly(err))
if pgn is None:
    st.caption("Enter a valid game above to enable the button.")


# ------------------------------------------------------------------ results
def build_frames(moves):
    """One frame per position (0 = start): FEN, last move, eval and White's win chance.

    The replay component steps through these in the browser, so moving between
    positions (arrow keys, slider) never triggers a Streamlit rerun.
    """
    b = chess.Board()
    frames = [{"ply": 0, "fen": b.board_fen(), "san": None, "side": None, "uci": None,
               "eval_cp": 0, "wp": 0.5, "loss": 0.0, "label": "ok"}]
    for m in moves:
        mv = b.push_san(m["san"])
        frames.append({"ply": m["ply"], "fen": b.board_fen(), "san": m["san"],
                       "side": m["side"], "uci": mv.uci(), "eval_cp": m["eval_cp"],
                       "wp": m["white_win_prob"], "loss": m["loss"], "label": m["label"]})
    return frames


r = st.session_state.get("result")
if r:
    st.subheader("3. Results")
    for w in r["warnings"]:
        st.warning(w)
    cols = st.columns(len(r["players"]))
    for col, p in zip(cols, r["players"]):
        delta = None
        if p["actual_rating"] is not None:
            delta = f"{p['predicted_rating'] - p['actual_rating']:+.0f} vs rating in file ({p['actual_rating']})"
        col.metric(f"{p['name']} ({p['side'].capitalize()})",
                   f"{p['predicted_rating']:.0f}", delta, delta_color="off")

    moves = r["moves"]
    st.markdown("##### Who is winning, move by move")
    st.caption("Higher means better for White. Dots mark moves that gave away winning chances.")
    df = pd.DataFrame({"Move": [m["ply"] for m in moves],
                       "White's win chance (%)": [m["white_win_prob"] * 100 for m in moves],
                       "Played": [m["san"] for m in moves],
                       "Quality": [m["label"] for m in moves]})
    line = alt.Chart(df).mark_line().encode(
        x="Move:Q", y=alt.Y("White's win chance (%):Q", scale=alt.Scale(domain=[0, 100])))
    dots = alt.Chart(df[df["Quality"] != "ok"]).mark_circle(size=90).encode(
        x="Move:Q", y="White's win chance (%):Q", tooltip=["Move", "Played", "Quality"],
        color=alt.Color("Quality:N", scale=alt.Scale(
            domain=["inaccuracy", "mistake", "blunder"], range=["#4c78a8", "#f58518", "#e45756"])))
    st.altair_chart(line + dots, use_container_width=True)

    summ = r["move_summary"]
    st.write(" · ".join(
        f"**{s.capitalize()}**: {summ[s]['blunder']} blunders, {summ[s]['mistake']} mistakes, "
        f"{summ[s]['inaccuracy']} inaccuracies" for s in ("white", "black")))
    a, g = r["analysis"], r["game"]
    st.caption(f"{g['plies']} half-moves analysed in {a['seconds']} s by {a['engine']} "
               f"(depth {a['depth']}). Rating model took {r['model']['seconds']} s.")

    st.markdown("##### Replay")
    st.caption("Step with the ← → arrow keys, the buttons or the slider. The card always "
               "shows the side that is ahead.")
    game_replay(frames=build_frames(moves), pieces=piece_svgs(), sig=r["request_id"],
                key="replay", default=None)
    with st.expander("Move list"):
        parts = []
        for m in moves:
            text = m["san"] + MARKS.get(m["label"], ("", ""))[0]
            if m["label"] in MARKS:
                text = f":{MARKS[m['label']][1]}[{text}]"
            parts.append((f"{(m['ply'] + 1) // 2}. " if m["side"] == "white" else "") + text)
        st.markdown(" ".join(parts))
        st.caption("?! inaccuracy · ? mistake · ?? blunder")