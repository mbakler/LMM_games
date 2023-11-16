import torch
from peft import PeftModel
from players.image_module import ImageModule
from language_module import LanguageModule
class TaxiPlayer():
    def __init__(self, args) -> None:
      self.image_module = ImageModule(args)
      # prompts for different baseline types
      self.prompts = args.prompt_path
      if args.language_model:
          self.language_module = LanguageModule(args)

    def act(self, image, observation):
      if self.args.player_type == "baseline": # Baseline is ask directly the image module
        action = self.image_module.query_image(image, observation, self.prompts["baseline"])
      elif self.args.player_type == "describe-act": # Get the LMM to describe and LLM to act
        image_output = self.image_module.query_image(image, observation, self.prompts["describe-act"]["decribe"])
        action = self.language_module.synthesise_answer(image_output, observation, self.prompts["describe-act"]["act"], self.args)
      return action

