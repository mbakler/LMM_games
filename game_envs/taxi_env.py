import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image
class Taxi_game():
    """
    https://gymnasium.farama.org/environments/toy_text/taxi/

    """
    def __init__(self) -> None:
        self.env = gym.make('Taxi-v3', render_mode="rgb_array")

    def run_episode(self, player, seed):
        observation, info = self.env.reset(seed=seed)
        observation = self.observation_decoder(observation)
        print(observation)
        rewards = []
        logs = []

        for _ in range(20):
            current_image = self.env.render()
            image_object = Image.fromarray(current_image).convert("RGB")
            # show the image to the player
            plt.imshow(image_object)
            #response_string = player.act(image_object, observation)
            #action = self.parse_action(response_string)
            #retry = 5
            #while action is None and retry > 0:
            #  response_string = player.act(image_object, observation)
            #  action = self.parse_action(response_string)
            #  retry -= 1
            #if action is None:
            #  return None, None
            action = self.env.action_space.sample()  # agent policy that uses the observation and info
            new_observation, reward, terminated, truncated, info = self.env.step(action)
            #logs.append({"observation": observation,
            #             "image": image_object,
            #             "response_string": response_string,
            #            "action": action,
            #            "reward": reward,
            #            "terminated": terminated,
            #            "truncated": truncated,
            #            "info": info})
            observation = new_observation
            rewards.append(reward)
            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()
        return logs, rewards

    def parse_action(self, response_string):
        patterns = {
            0: ["DRIVE DOWN"],
            1: ["DRIVE UP"],
            2: ["DRIVE RIGHT"],
            3: ["DRIVE LEFT"],
            4: ["PICKUP"],
            5: ["DROPOFF"],
        }
        last_action = (-1, -1)
        for key, values in patterns.items():
            for value in values:
                if value.lower() in response_string.lower():
                    # get the index of the last action mentioned (sometimes LLMs are dumb and output actions with wrong order or casing)
                    index = response_string.lower().index(value.lower())
                    if index > last_action[1]:
                        last_action = (key, index)
        if last_action[0] != -1:
            return last_action[0]
        return None
    
    def observation_decoder(self, state):
        """
        Decode the state to a human readable string
        """
        for taxi_row in range(5):
            for taxi_col in range(5):
                for pass_loc in range(5):
                    for dest_idx in range(4):
                        state_idx = ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx
                        if state_idx == state:
                            return {"taxi_row": taxi_row,
                                    "taxi_col": taxi_col,
                                    "pass_loc": pass_loc,
                                    "dest_idx": dest_idx}
                            
