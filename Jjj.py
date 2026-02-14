from openai import OpenAI
import torch


r = input('Enter your request: ')


client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = "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",  
    )


completion = client.chat.completions.create(
    model = "tngtech/deepseek-r1t2-chimera:free",
    messages = [
        {"role": "system",
         "content": "Отвечай двумя словами"
         },
        {"role": "user",
         "content": r
         },
      ],
     )

language = "ru"
model_id = "v5_ru"
speaker = "xenia"
sample_rate = 4800

device = torch.device("cpu")

torch.hub.set_dir("D:\\Projects\\Silero")

model, example_text = torch.hub.load(repo_or_dir = "snakers4/silero-models",
                                     model='silero_tts',
                                     language = language,
                                     speaker = model_id)

model.to(device)


audio = model.apply_tts( text = completion.choices[0].message.content,
                         speaker = speaker,
                         sample_rae = sample_rate
    )

print (torch.hub.get_dir())
