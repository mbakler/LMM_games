from LLaVa.model.builder import load_pretrained_model
from LLaVa.utils import disable_torch_init
from LLaVa.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import re
from LLaVa.conversation import conv_templates, SeparatorStyle
from LLaVa.mm_utils import (
    tokenizer_image_token,
    KeywordsStoppingCriteria,
    process_images,
)
import torch
from peft import PeftModel

class Player_LLaVa():
    def __init__(self, args) -> None:
      self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
          args.model_path, args.model_base, args.model_name
      )
      self.args = args
      if args.lora_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                args.lora_path,
            )
      # prompts for different baseline types
      self.prompts = {
          "baseline":"This is a picture of a game. Your goal is to output the correct next action of the taxi depending on the picture and the situation. If there is a pixelated character in the picture, you need to drive the taxi to the character. If your taxi is on the same square as the character you need to pick up the character. If there is no character in the picture, you need to drive to the building square. If your taxi is on the building square, you need to drop off the character. Analyse the picture thoroughly and output the correct next action that the the taxi should take in this situation. The allowed actions that the taxi can take are DRIVE LEFT (taxi should move 1 square to the left), DRIVE RIGHT (taxi should move 1 step to the right), DRIVE UP (taxi should move 1 square up), DRIVE DOWN (taxi should move 1 square down), PICKUP (taxi should pickup the character), DROPOFF (taxi should drop off the character. Return only the next action for the taxi and nothing else"
          }

    def act(self, image, observation):
      if self.args.player_type == "baseline":
        self.args.query = self.prompts["baseline"]
        images_tensor = process_images(
                                          [image],
                                          self.image_processor,
                                          self.model.config
                                      ).to(self.model.device, dtype=torch.float16)
        text_output = self._get_llava_output(images_tensor, self.args, self.model, self.tokenizer, self.image_processor,)
        #
        return text_output
    def _get_llava_output(self, image_tensor, args, model, tokenizer, image_processor):
      # Model
      disable_torch_init()

      qs = args.query
      image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
      if IMAGE_PLACEHOLDER in qs:
          if model.config.mm_use_im_start_end:
              qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
          else:
              qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
      else:
          if model.config.mm_use_im_start_end:
              qs = image_token_se + "\n" + qs
          else:
              qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

      if "llama-2" in args.model_name.lower():
          conv_mode = "llava_llama_2"
      elif "v1" in args.model_name.lower():
          conv_mode = "llava_v1"
      elif "mpt" in args.model_name.lower():
          conv_mode = "mpt"
      else:
          conv_mode = "llava_v0"

      if args.conv_mode is not None and conv_mode != args.conv_mode:
          print(
              "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                  conv_mode, args.conv_mode, args.conv_mode
              )
          )
      else:
          args.conv_mode = conv_mode

      conv = conv_templates[args.conv_mode].copy()
      conv.append_message(conv.roles[0], qs)
      conv.append_message(conv.roles[1], None)
      prompt = conv.get_prompt()

      input_ids = (
          tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
          .unsqueeze(0)
          .cuda()
      )

      stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
      keywords = [stop_str]
      stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

      with torch.inference_mode():
          output_ids = model.generate(
              input_ids = input_ids,
              images=image_tensor,
              do_sample=True if args.temperature > 0 else False,
              temperature=args.temperature,
              top_p=args.top_p,
              num_beams=args.num_beams,
              max_new_tokens=args.max_new_tokens,
              use_cache=True,
              stopping_criteria=[stopping_criteria],
          )

      input_token_len = input_ids.shape[1]
      n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
      if n_diff_input_output > 0:
          print(
              f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
          )
      outputs = tokenizer.batch_decode(
          output_ids[:, input_token_len:], skip_special_tokens=True
      )[0]
      outputs = outputs.strip()
      if outputs.endswith(stop_str):
          outputs = outputs[: -len(stop_str)]
      outputs = outputs.strip()
      return outputs