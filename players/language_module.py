import os
import requests
import time
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class LanguageModule():
    def __init__(self, language_model_type) -> None:
      # prompts for different baseline types
        if "openai" in language_model_type:
            self.language_module = OpenAI()
      

    def synthesise_answer(self, description, observation, prompt, args):
        args.prompt = prompt.format(image_description=description)
        text_output = self.language_module.synthesise_answer(args)
        return text_output
      
class OpenAI():
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("OpenAI API key is not set")

    def synthesise_answer(self, args):
      # Model
        temperature = args.get("llm_temperature", 0)
        top_p = args.get("llm_top_p", 1)
        frequency_penalty = args.get("llm_frequency_penalty", 0)
        presence_penalty = args.get("llm_presence_penalty", 0)
        model = args.get("llm_model", "gpt-3.5-turbo-1106")
        params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 512,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        messages=[
                    {
                        "role": "system",
                        "content": "You are a clever language model, who outputs actions using image descriptions and action instructions"
                    },
                    {
                        "role": "user",
                        "content": args.prompt
                    }
                ]
        params["messages"] = messages
        counter = 0
        choice = None
        # initiate response so exception logic doesnt error out when checking for error in response
        response = {}
        while counter < 5:
            try:
                openai_headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                response = requests.post(
                    OPENAI_URL, headers=openai_headers, json=params, timeout=50
                )
                response = response.json()
                choice = response["choices"][0]["message"]["content"].strip("'")
                break
            except Exception:
                if ("error" in response and 
                    "code" in response["error"] and 
                    response["error"]["code"] == 'invalid_api_key'):
                    raise Exception(f"The supplied OpenAI API key {self.api_key} is invalid")

                time.sleep(1 + 3 * counter)
                counter += 1
                continue
            
        if not choice:
            raise Exception("OpenAI API failed to generate a response")
        return choice