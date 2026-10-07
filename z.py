path = "/home/oc/agentic_ml/dataset/train.csv"

import pandas as pd
from tools.data.leakage import detect_leakage

from tools.data.profile import inspect_dataset, profile_columns, detect_target
from tools.data.quality import check_data_quality
from tools.data.leakage import detect_leakage

df = pd.read_csv(path)

# inspection = inspect_dataset(df)
# profile = profile_columns(df)
# quality = check_data_quality(df, profile=profile)
# target_result = detect_target(df, profile=profile, objective="predict customer churn")


# print(target_result)
# print(quality)
# print(inspection)
# print(profile)



from tests.integration.test_llm_providers import TestAnthropicProvider, TestGeminiProvider
from dotenv import load_dotenv
load_dotenv()
test_client = TestGeminiProvider()
test_client._setup_client()
test_client.test_key_is_valid_simple_completion()




from agent.gemini_client import GeminiLiteLLMClient
from dotenv import load_dotenv
load_dotenv()

# import os
# client = GeminiLiteLLMClient(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url=os.getenv("LITELLM_GEMINI_BASE_URL"),
# )
# response =client.create(
#     model="genuin/vid-gen",
#     max_tokens=1000,
#     system="You are a helpful assistant.",
#     tools=[],
#     messages=[{"role": "user", "content": "What is the capital of France?"}],
# )   


# print(response.content)





from openai import OpenAI

import os
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=f"{os.getenv("LITELLM_BASE_URL")}",
)


response = client.chat.completions.create(
    model="genuin/clipit",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    tools=[],
)

print(response.choices[0].message.content)

# print(response.choices[0].message.content)