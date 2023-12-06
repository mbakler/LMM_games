# LMM_games

A very work-in-progress sideproject to get multimodal LLMs (like Llava) to play games. Currently in the progress of testing it out on OpenAIs taxi env.

Early results of some tests

* Llava and GPT4-Vision both have difficulties in understanding directions (i.e is the person left or right from the taxi) and then taking the correct action -- likely need some sort of grounding (bounding-box approach to fix coordinates) or SFT/DPO finetuning on LLM portion (assuming this isn't an artifact from LLava clip model embeddings which are usually frozen).
