"""In-app user manual (source for docs/USER_MANUAL.md)."""
import streamlit as st

st.set_page_config(page_title="Help", page_icon="❓")
st.title("❓ How to use this tool")
st.markdown("""
**What it does.** You give it one chess game. It checks every move with a chess engine
and estimates the rating of each player.

### Step 1: give it a game
- **Paste PGN.** Copy the game text from a site such as Lichess or Chess.com and paste it.
  *Insert example game* fills in a sample.
- **Upload a file.** Choose a `.pgn` or `.txt` file. If it holds several games, pick one.
- **Play it on a board.** Choose each move from the list, use *Undo* to take one back, and
  say how the game ended.

A green message means the game is fine. A red one tells you what to fix.

### Step 2: press *Analyse game*
Analysis takes a few seconds, and up to a minute for long games.

### Step 3: read the results
- **Estimated rating** per player, next to the rating in the file if there was one.
  Treat it as an estimate, not an exact number.
- **The graph** shows who is winning after each move. Dots mark moves that threw away
  winning chances: blue = inaccuracy, orange = mistake, red = blunder.
- **Replay** lets you step through the game.

### Good to know
- Very short games and games with an unknown result give less reliable estimates.
- Only standard chess is supported, one game at a time.
- A red *service* message in the side bar means the server is down. Try again later.
""")