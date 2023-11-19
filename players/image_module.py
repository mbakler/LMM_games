from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import re
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.mm_utils import (
    tokenizer_image_token,
    KeywordsStoppingCriteria,
    process_images,
)
import torch
from peft import PeftModel

class ImageModule():
    def __init__(self, args) -> None:
        self.args = args
      # prompts for different baseline types
        if "llava" in args.model_name.lower():
            self.image_module = LLava(args)
      

    def query_image(self, image, observation, query):
        self.args.query = query
        text_output = self.image_module.get_image_output(image, self.args)
        return text_output
      
class LLava():
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

    def get_image_output(self, image, args):
      # Model
      disable_torch_init()
      images_tensor = process_images(
                                          [image],
                                          self.image_processor,
                                          self.model.config
                                      ).to(self.model.device, dtype=torch.float16)
      qs = args.query
      image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
      if IMAGE_PLACEHOLDER in qs:
          if self.model.config.mm_use_im_start_end:
              qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
          else:
              qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
      else:
          if self.model.config.mm_use_im_start_end:
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
          tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
          .unsqueeze(0)
          .cuda()
      )

      stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
      keywords = [stop_str]
      stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

      with torch.inference_mode():
          output_ids = self.model.generate(
              input_ids = input_ids,
              images=images_tensor,
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
      outputs = self.tokenizer.batch_decode(
          output_ids[:, input_token_len:], skip_special_tokens=True
      )[0]
      outputs = outputs.strip()
      if outputs.endswith(stop_str):
          outputs = outputs[: -len(stop_str)]
      outputs = outputs.strip()
      return outputs