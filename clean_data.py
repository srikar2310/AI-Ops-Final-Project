import os
import chess.pgn

INPUT_BASE = "dataset"
CLEAN_BASE = "dataset_clean"

splits = [("train", 10000), ("val", 5000), ("test", 5000)]

print("Starting full dataset cleaning...")

for split, shard_size in splits:
  input_dir = os.path.join(INPUT_BASE, split)
  output_dir = os.path.join(CLEAN_BASE, split)

  if not os.path.exists(input_dir):
    print(f"Directory {input_dir} not found. Skipping.")
    continue

  os.makedirs(output_dir, exist_ok=True)
  out_shard_idx = 1
  out_count = 0
  total_cleaned = 0


  def open_output_shard(idx):
    return open(
        os.path.join(output_dir, f"{split}_shard_{idx:03d}.pgn"),
        "w",
        encoding="utf-8",
    )


  out_file = open_output_shard(out_shard_idx)
  print(f"Processing {split} shards...")

  for filename in sorted(os.listdir(input_dir)):
    if not filename.endswith(".pgn"):
      continue
    filepath = os.path.join(input_dir, filename)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
      while True:
        game = chess.pgn.read_game(f)
        if game is None:
          break

        # Keep only WhiteElo, BlackElo, and Result headers
        w_elo = game.headers.get("WhiteElo", "")
        b_elo = game.headers.get("BlackElo", "")
        result = game.headers.get("Result", "*")

        game.headers.clear()
        if w_elo:
          game.headers["WhiteElo"] = w_elo
        if b_elo:
          game.headers["BlackElo"] = b_elo
        game.headers["Result"] = result

        # Strip all comments (engine evals, clocks) from move nodes
        node = game
        while node is not None:
          node.comment = ""
          node = node.next()

        out_file.write(str(game) + "\n\n")
        total_cleaned += 1
        out_count += 1

        # Maintain shard size limits
        if out_count >= shard_size:
          out_file.close()
          out_shard_idx += 1
          out_file = open_output_shard(out_shard_idx)
          out_count = 0

  out_file.close()
  print(
      f"Successfully cleaned {total_cleaned} {split} games -> saved in"
      f" '{output_dir}/'"
  )

print("\n--- All Games Cleaned Successfully ---")