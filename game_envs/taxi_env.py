import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image
# import sign function
from numpy import sign

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
        orientation_logs = {}
        for _ in range(20):
            current_image = self.env.render()
            image_object = Image.fromarray(current_image).convert("RGB")
            # show the image to the player
            #plt.imshow(image_object)
            observation = self.observation_decoder(observation)
            print(observation)
            response_string = player.act(image_object, observation)
            #accurate, axis = self.evaluate_orientation(observation, response_string, target = "passenger")
            #if axis not in orientation_logs.keys():
            #  orientation_logs[axis] = []
            #orientation_logs[axis].append(accurate)
            #print(f"Passenger: accuracy {accurate}, axis {axis}")
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
                            
    def reward_model(observation, action, target = "pass"):
      action_dict = {}
      if "DRIVE-UP" in action:
        action_dict = {"action_col": 0, "action_row": -1}
      elif "DRIVE-DOWN" in action:
        action_dict = {"action_col": 0, "action_row": 1}
      elif "DRIVE-LEFT" in action:
        action_dict = {"action_col": -1, "action_row": 0}
      elif "DRIVE-RIGHT" in action:
        action_dict = {"action_col": 1, "action_row": 0}
      
      if "action_col" not in action_dict.keys():
        return -1


      target_row = observation[f"{target}_row"]
      target_col = observation[f"{target}_col"]
      taxi_row = observation["taxi_row"]
      taxi_col = observation["taxi_col"]
      # check which cuadrant is the target and the taxi
      quadrants = {0:{"row":[0,1], "col":[0, 1]},
                    1:{"row": [0,1], "col": [2,3,4]}, 
                    2:{"row": [3,4], "col": [0]}, 
                    3:{"row": [3,4], "col": [1,2]}, 
                    4:{"row": [3,4], "col": [3,4]}, 
                    }
      target_quadrant = get_quadrants(quadrants, target_row, target_col)
      # if youre on the 3rd row, you can only move towards target col
      # or enter the correct quadrant
      if taxi_row == 2:
          if action_dict["action_col"] != 0:
            if sign(action_dict["action_col"]) == sign(target_col - taxi_col):
              return 1
            else:
              return 0
          else:
            taxi_row += action_dict["action_row"]
            taxi_quad = get_quadrants(quadrants, taxi_row, taxi_col)
            if taxi_quad == target_quadrant:
              return 1
            else:
              return 0
      
      # otherwise check if youre in the correct quadrant
      taxi_quad = get_quadrants(quadrants, taxi_row, taxi_col)
      if taxi_quad == target_quadrant:
        if action_dict["action_col"] != 0:
          if sign(action_dict["action_col"]) == sign(target_col - taxi_col):
            return 1
          else:
            return 0
        else:
          if sign(action_dict["action_row"]) == sign(target_row - taxi_row):
            return 1
          else:
            return 0
      else:
         # either move towards the correct column within the same quadrant
         # or move towards the 3rd row
        if action_dict["action_col"] != 0:
           # check that the new column is in the same quadrant
          if not get_quadrants(quadrants, taxi_row, taxi_col + action_dict["action_col"]) == taxi_quad:
            return 0
          # check that we're moving towards the target
          if sign(action_dict["action_col"]) == sign(target_col - taxi_col):
            return 1
          else:
            return 0
  
        else:
          if sign(action_dict["action_row"]) == sign(target_row - taxi_row):
            return 1
          else:
            return 0

      


    def evaluate_orientation(self, observation, response_string, target = "passenger"):
        """
        Evaluate the true and predicted orientation of the taxi to the passenger and the destination
        """
        observation = self.observation_decoder(observation)
        #print(observation)
        pass_row = observation["pass_row"]
        pass_col = observation["pass_col"]
        taxi_row = observation["taxi_row"]
        taxi_col = observation["taxi_col"]
        dest_row = observation["dest_row"]
        dest_col = observation["dest_col"]


        if target == "passenger":
            eval_row = pass_row
            eval_col = pass_col
        elif target == "destination":
            eval_row = dest_row
            eval_col = dest_col

        if "up" in response_string.lower() or "down" in response_string.lower():
          if taxi_row < eval_row:
              row_truth =  "down"
              row_false = "up"
          elif taxi_row > eval_row:
              row_truth =  "up"
              row_false = "down"
          else:
            return not ("up" in response_string.lower() or "down" in response_string.lower()), "equal_y"
          return row_truth in response_string.lower(), "y"
        elif "left" in response_string.lower() or "right" in response_string.lower():
          if taxi_col > eval_col:
            col_truth = "left"
            col_false = "right"
          elif taxi_col < eval_col:
            col_truth = "right"
            col_false = "left"
          else:
            return not ("left" in response_string.lower() or "right" in response_string.lower()), "equal_x"
          return col_truth in response_string.lower(), "x"
        return False, "undefined"
    

def get_quadrants(quadrants, row, col):
  for key, value in quadrants.items():
    if row in value["row"] and col in value["col"]:
      return key