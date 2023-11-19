import torch
from peft import PeftModel
from players.image_module import ImageModule
from players.language_module import LanguageModule
import json
class TaxiPlayer():
    def __init__(self, args) -> None:
      self.args = args
      self.image_module = ImageModule(args)
      # read in the prompts
      with open(args.prompt_path, "r") as f:
        self.prompts = json.load(f)

      if args.language_model:
          self.language_module = LanguageModule(args.language_model)

    def act(self, image, observation):
      if self.args.player_type == "baseline": # Baseline is ask directly the image module
        action = self.image_module.query_image(image, observation, self.prompts["baseline"])
      elif self.args.player_type == "describe-act": # Get the LMM to describe and LLM to act
        image_output = self.image_module.query_image(image, observation, self.prompts["describe-act"]["describe"])
        action = self.language_module.synthesise_answer(image_output, observation, self.prompts["describe-act"]["act"], self.args)
      return action

