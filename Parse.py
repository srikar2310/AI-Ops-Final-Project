import math
import os
import random
import chess.pgn

input_path = "/home/srikar-j-v/courses/DA3408/Final Project/games/lichess_db_standard_rated_2017-01.pgn"

os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/val", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

TARGET_TRAIN = 100000
TARGET_VAL = 20000
TARGET_TEST = 20000
TOTAL_TARGET = TARGET_TRAIN + TARGET_VAL + TARGET_TEST  # 140,000

# 95% Thresholds for Early Stopping
STOP_TRAIN = int(TARGET_TRAIN * 0.95)  # 95,000
STOP_VAL = int(TARGET_VAL * 0.95)  # 19,000
STOP_TEST = int(TARGET_TEST * 0.95)  # 19,000

MU = 1800
SIGMA = 220

# Capped at 2500 max for historical 2017 database accuracy
bin_edges = list(range(500, 2600, 100))
num_bins = len(bin_edges) - 1


def get_normal_weight(x):
  return (1 / (SIGMA * math.sqrt(2 * math.pi))) * math.exp(
      -((x - MU) ** 2) / (2 * SIGMA**2)
  )


bin_weights = [
    get_normal_weight((bin_edges[i] + bin_edges[i + 1]) / 2)
    for i in range(num_bins)
]
total_weight = sum(bin_weights)
bin_probabilities = [w / total_weight for w in bin_weights]
bin_targets = [math.ceil(p * TOTAL_TARGET) for p in bin_probabilities]
bin_collected = [0] * num_bins

train_collected = 0
val_collected = 0
test_collected = 0
scanned = 0

SHARD_SIZE_TRAIN = 10000
SHARD_SIZE_VAL_TEST = 5000


def open_train_shard(idx):
  return open(
      f"dataset/train/train_shard_{idx:03d}.pgn", "w", encoding="utf-8"
  )


def open_val_shard(idx):
  return open(f"dataset/val/val_shard_{idx:03d}.pgn", "w", encoding="utf-8")


def open_test_shard(idx):
  return open(f"dataset/test/test_shard_{idx:03d}.pgn", "w", encoding="utf-8")


train_shard_idx, val_shard_idx, test_shard_idx = 1, 1, 1
train_file = open_train_shard(train_shard_idx)
val_file = open_val_shard(val_shard_idx)
test_file = open_test_shard(test_shard_idx)

train_in_shard, val_in_shard, test_in_shard = 0, 0, 0

print(
    "\nStarting PGN parsing with 2017 rating cap and 95% early stopping..."
)

with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in:
  while True:
    # Early stop once 95% of targets are fulfilled
    if (
        train_collected >= STOP_TRAIN
        and val_collected >= STOP_VAL
        and test_collected >= STOP_TEST
    ):
      print(
          "\nReached 95% dataset quota threshold! Stopping early to save"
          " time."
      )
      break

    game = chess.pgn.read_game(f_in)
    if game is None:
      print("Reached end of file.")
      break

    scanned += 1
    headers = game.headers

    event = headers.get("Event", "")
    w_elo_str = headers.get("WhiteElo", "?")
    b_elo_str = headers.get("BlackElo", "?")
    time_control = headers.get("TimeControl", "-")

    if not (w_elo_str.isdigit() and b_elo_str.isdigit()):
      continue
    if "Rated" not in event:
      continue

    w_elo = int(w_elo_str)
    b_elo = int(b_elo_str)

    # Check if EITHER player fits an open bin that needs filling
    target_bin_idx = -1
    players = [w_elo, b_elo]
    random.shuffle(players)

    for elo in players:
      for i in range(num_bins):
        if (
            bin_edges[i] <= elo < bin_edges[i + 1]
            and bin_collected[i] < bin_targets[i]
        ):
          target_bin_idx = i
          break
      if target_bin_idx != -1:
        break

    if target_bin_idx == -1:
      continue

    # Additional filters (Bullet check & Move count)
    ply_count = sum(1 for _ in game.mainline())
    move_count = ply_count // 2

    is_bullet = False
    if "+" in time_control:
      try:
        base, inc = map(int, time_control.split("+"))
        if (base + 40 * inc) < 180:
          is_bullet = True
      except ValueError:
        pass
    elif time_control.isdigit() and int(time_control) < 180:
      is_bullet = True

    if is_bullet or move_count <= 10:
      continue

    # Assign split (respecting individual targets so one split doesn't overfill)
    choices, weights = [], []
    if train_collected < TARGET_TRAIN:
      choices.append("train")
      weights.append(5)
    if val_collected < TARGET_VAL:
      choices.append("val")
      weights.append(1)
    if test_collected < TARGET_TEST:
      choices.append("test")
      weights.append(1)

    if not choices:
      break

    split = random.choices(choices, weights=weights, k=1)[0]
    game_str = str(game) + "\n\n"

    bin_collected[target_bin_idx] += 1

    if split == "train":
      train_file.write(game_str)
      train_collected += 1
      train_in_shard += 1
      if train_in_shard >= SHARD_SIZE_TRAIN and train_collected < TARGET_TRAIN:
        train_file.close()
        train_shard_idx += 1
        train_file = open_train_shard(train_shard_idx)
        train_in_shard = 0
    elif split == "val":
      val_file.write(game_str)
      val_collected += 1
      val_in_shard += 1
      if val_in_shard >= SHARD_SIZE_VAL_TEST and val_collected < TARGET_VAL:
        val_file.close()
        val_shard_idx += 1
        val_file = open_val_shard(val_shard_idx)
        val_in_shard = 0
    elif split == "test":
      test_file.write(game_str)
      test_collected += 1
      test_in_shard += 1
      if test_in_shard >= SHARD_SIZE_VAL_TEST and test_collected < TARGET_TEST:
        test_file.close()
        test_shard_idx += 1
        test_file = open_test_shard(test_shard_idx)
        test_in_shard = 0

    if scanned % 10000 == 0:
      print(
          f"Scanned: {scanned} | Train: {train_collected}/{TARGET_TRAIN} | Val:"
          f" {val_collected}/{TARGET_VAL} | Test:"
          f" {test_collected}/{TARGET_TEST}"
      )

train_file.close()
val_file.close()
test_file.close()

print("\n--- Processing Complete ---")
print(
    f"Final Collected -> Train: {train_collected}, Val: {val_collected},"
    f" Test: {test_collected}"
)