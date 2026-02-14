from openai import OpenAI
import torch


r = input('Enter your request: ')


client = OpenAI(
    base_url = "https://openrouter.ai/api/v1", #URL LINK (do not change it)
    api_key = "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",  #API KEY
    )


completion = client.chat.completions.create(
    model = "tngtech/deepseek-r1t2-chimera:free", #MODEL ID
    messages = [
        {"role": "system",
         "content": "Speak like a teacher" #OUR PROMPT
         },                                #IMPORTANT! IF U WROTE YOUR PROMPT ON ENGLISH, BUT YOU TYPED A RUSSIAN SENTENCE (OR IN REVERSE) IT WILL CAUSE A PROBLEM WITH SILERO!!!!!!! 
        {"role": "user",
         "content": r
         },
      ],
     )
#SILERO_TTS
language = "ru"
model_id = "v5_ru" # MODEL ID OF SILERO_TTS
speaker = "xenia" #SPEAKER OF SILERO_TTS
sample_rate = 4800

device = torch.device("cpu") 

torch.hub.set_dir("D:\\Projects\\Silero")     #PATH TO DOWNLOAD SILERO_TTS

model, example_text = torch.hub.load(repo_or_dir = "snakers4/silero-models", # DESPITE SILERO YOU CAN USE QWen3_TTS OR Google_TTS
                                     model='silero_tts',
                                     language = language,
                                     speaker = model_id)

model.to(device)


audio = model.apply_tts( text = completion.choices[0].message.content, # IN ORDER TO VOICE THE MESSAGE FROM AI
                         speaker = speaker,
                         sample_rae = sample_rate
    )

print (torch.hub.get_dir())

#IF YOU GET AN ERROR 429 JUST RETRY RUNNING YOUR CODE!!!!!!!
