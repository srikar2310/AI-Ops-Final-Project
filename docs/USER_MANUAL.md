# User Manual: Chess Rating Estimator

## What it does

You give it one chess game. It checks every move with a chess engine and estimates the rating of each player.

It is useful when you want to know roughly how strong a game was played, for example a game against an unrated opponent or an over-the-board game you typed in.

## Opening the app

If someone has already started the app for you, open your web browser and go to the address they gave you. On the computer running it, that is:

**http://localhost:8501**

You should see the main page with a box for your game.

> **For the person setting it up:** in the project folder run `docker compose up -d --build`, wait about a minute until `docker compose ps` shows every service as running, then open http://localhost:8501. Stop it with `docker compose down`.

## Step 1: give it a game

You can give the app a game in three ways. Pick one.

- **Paste PGN.** Copy the game text from a site such as Lichess or Chess.com and paste it. *Insert example game* fills in a sample, so you can try the tool without a game of your own. 
  *Where to find the PGN:* on Lichess, open the game, click **Share & export**, and copy the PGN. On Chess.com, open the game, click the **Share** icon, choose **PGN**, and copy it.

- **Upload a file.** Choose a `.pgn` or `.txt` file. If it holds several games, pick one. 

- **Play it on a board.** Choose each move from the list, use *Undo* to take one back, and say how the game ended.

A green message means the game is fine. A red one tells you what to fix.

## Step 2: press *Analyse game*

Analysis takes a few seconds, and up to a minute for long games. Please don't press the button again while it is working.

## Step 3: read the results

- **Estimated rating** per player, next to the rating in the file if there was one. Treat it as an estimate, not an exact number. On the test games, the estimate was within 200 points of the real rating about 6 times out of 10, and off by about 180 points on average.
- **The graph** shows who is winning after each move. Dots mark moves that threw away winning chances: blue = inaccuracy, orange = mistake, red = blunder.
- **Replay** lets you step through the game.

## Good to know

- Very short games and games with an unknown result give less reliable estimates.
- Only standard chess is supported, one game at a time.
- A red *service* message in the side bar means the server is down. Try again later.
- The estimate is based only on how well the moves were played. It does not know the time control or how much time each player had.
- Estimates for very strong players (2200 and above) tend to come out too low, and for beginners (below 1200) too high, often by around 400 points.
- Your games are not saved.

## Troubleshooting

<!-- TODO before submitting: make the wording in the first column match the app's actual error messages, and replace the screenshot placeholders. -->

| What you see | What it means | What to do |
|---|---|---|
| Red message about the game text | The text is not a valid chess game, or has an illegal move. | Copy the PGN again from the site's export button; make sure you copied the whole game. Try *Insert example game* to check the app itself works. |
| Red message that the game is too short | There are too few moves to judge. | Use a longer game. |
| The file is not accepted | It is not a `.pgn` or `.txt` file, or it is empty. | Export the game again as PGN. |
| Red *service* message in the side bar | The analysis server is not reachable. | Wait a minute and refresh the page. If it stays, ask the person who runs the app to restart it. |
| Analysis runs for more than a minute | The game is very long or the server is busy. | Wait, or try again later. |
| The page does not open at all | The app is not running. | Ask the person who runs it to start it (see *Opening the app*). |

## For curious users: the other pages

- **Pipeline Console** shows how the model was built: the data and training steps, past training runs, and how many succeeded or failed. It also links to the monitoring dashboards. You don't need it to estimate a rating.
- **Help** is a short version of this manual inside the app.

## Glossary

- **PGN:** the standard text format for a chess game, a list of moves plus a few lines like who played and the result.
- **Rating (Elo):** a number for playing strength. Beginners are often below 1000; club players 1400 to 2000; masters above 2200.
- **Inaccuracy, mistake, blunder:** moves that lose a small, medium or large share of the chances of winning.
