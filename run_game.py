import argparse
import os
from LLaVa.mm_utils import get_model_name_from_path
from players.player_llava import Player_LLaVa
from game_envs.taxi_env import Taxi_game
# parse arguments
parser = argparse.ArgumentParser(description='Description of your program') # example "/content/LLaVA/LLaVA-RLHF-13b-v1.5-336/sft_model"
parser.add_argument('-model_path', help='Path of the model', required=True)
parser.add_argument('-baseline', help='Player type', default="baseline", required=False)
parser.add_argument('-lora_path', help='Path of the lora model', required=False) #"LLaVA/LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model"
parser.add_argument('-model_base', help='Model base', required=False)
parser.add_argument('-conv_mode', help='Conv mode', required=False)
parser.add_argument('-sep', help='Separator', default= ",", required=False)
parser.add_argument('-temperature', help='Temperature', default=0.2 ,required=False)
parser.add_argument('-top_p', help='Top p', default=None, required=False)
parser.add_argument('-num_beams', help='Number of beams', default=1, required=False)
parser.add_argument('-max_new_tokens', help='Max new tokens', default=256, required=False)
args = parser.parse_args()
args.model_name = get_model_name_from_path(args.model_path) # example "llava-rlhf-13b-v1.5-336"


def play_game(game, player):
    seeds = [1, 2, 3, 4, 5]
    nr_of_plays = 5
    games = {seed: [] for seed in seeds}
    for seed in seeds:
        for _ in range(nr_of_plays):
            logs, rewards = game.run_episode(player, seed=seed)
            games[seed].append({"logs": logs, "rewards": rewards})
            print(rewards)
    return games

def run_simulation(args):
    # define game and player
    player = Player_LLaVa(args)
    game = Taxi_game()
    # finally play the game
    games = play_game(game, player)
    # save the games as a pkl object
    import pickle
    with open("games.pkl", "wb") as f:
        pickle.dump(games, f)