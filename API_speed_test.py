import os
import time

from openai import OpenAI
from pydub import AudioSegment



client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])






# randomly generate 10 english sentece in the prompt list
prompt_list = [
    "What is the main theme of Shakespeare's 'Macbeth'?",
    "How do the settings influence the plot in the novel 'To Kill a Mockingbird'?",
    "Why is the Great Depression a significant period in American history?",
    "What are the differences between metaphors and similes in English literature?",
    "How does photosynthesis contribute to the environment?",
    "What impact did the Industrial Revolution have on urban life?",
    "How do you identify the tone of a piece of writing?",
    "What are the constitutional checks and balances in the United States government?",
    "How does globalization affect local cultures?",
    "What are the ethical implications of artificial intelligence in society?"
]
result_list = []

# record the time needed
start_time = time.time()
for i in range(0, len(prompt_list)):
    current_time = time.time()
    prompt = prompt_list[i]
    result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    time_for_each_prompt = time.time() - current_time
    result_list.append(result)
    print(time_for_each_prompt)

time_taken = time.time() - start_time
print(result_list)
print(time_taken)
print("average api response time is :", time_taken/len(prompt_list))




completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
)


