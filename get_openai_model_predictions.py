from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="ft:gpt-4o-mini-2024-07-18:personal:trier-20-5-prompt-excluded:ANNq48Mj",
  messages= [
    {"role": "system", "content": "This assistant is trained to recall the next sequence of coordinates exactly, based on learned patterns from training data. It should output only the precise continuation of the input sequence without deviation."}, 
    {"role": "user", "content": "80 118 79 123 88 127 89 129 88 132 91 135 93 136 103 115 103 134 108 140 112 145 108 150 96 137 100 130 100 130 76 144 76 130 76 129 77 128 77 128"}
  ]
)
print(completion.choices[0].message)