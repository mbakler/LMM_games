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
        rewards = []
        logs = []

        for _ in range(20):
            current_image = self.env.render()
            image_object = Image.fromarray(current_image).convert("RGB")
            # show the image to the player
            plt.imshow(image_object)
            response_string = player.act(image_object, observation)
            x_true, y_true, x_false, y_false = self.evaluate_orientation(observation, response_string, target = "passenger")
            print(f"Passenger: x_true: {x_true}, y_true: {y_true}, x_false: {x_false}, y_false: {y_false}")
            action = self.parse_action(response_string)
            retry = 5
            while action is None and retry > 0:
              response_string = player.act(image_object, observation)
              action = self.parse_action(response_string)
              retry -= 1
            if action is None:
              return logs, rewards
            #action = self.env.action_space.sample()  # agent policy that uses the observation and info
            new_observation, reward, terminated, truncated, info = self.env.step(action)
            logs.append({"observation": observation,
                         "image": image_object,
                         "response_string": response_string,
                        "action": action,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info})
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
        colour_rows = [0, 0, 4, 4, -1]
        colour_cols = [0, 4, 0, 3, -1]
        for taxi_row in range(5):
            for taxi_col in range(5):
                for pass_loc in range(5):
                    for dest_idx in range(4):
                        state_idx = ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx
                        if state_idx == state:
                            return {"taxi_row": taxi_row,
                                    "taxi_col": taxi_col,
                                    "pass_row": colour_rows[pass_loc],
                                    "pass_col": colour_cols[pass_loc],
                                    "dest_row": colour_rows[dest_idx],
                                    "dest_col": colour_cols[dest_idx]}
                            

    def evaluate_orientation(self, observation, response_string, target = "passenger"):
        """
        Evaluate the true and predicted orientation of the taxi to the passenger and the destination
        """
        observation = self.observation_decoder(observation)
        pass_row = observation["pass_row"]
        pass_col = observation["pass_col"]
        taxi_row = observation["taxi_row"]
        taxi_col = observation["taxi_col"]
        dest_row = observation["dest_row"]
        dest_col = observation["dest_col"]
        if target == "passenger":
            eval_row = taxi_row
            eval_col = taxi_col
        elif target == "destination":
            eval_row = pass_row
            eval_col = pass_col
        
        if taxi_row < eval_row:
            row_truth =  "down"
            row_false = "up"
        elif taxi_row > eval_row:
            row_truth =  "up"
            row_false = "down"
        if eval_col > pass_col:
            col_truth = "left"
            col_false = "right"
        elif eval_col < pass_col:
            col_truth = "right"
            col_false = "left"
        return col_truth in response_string.lower(), row_truth in response_string.lower(), col_false in response_string.lower(), row_false in response_string.lower()