import os

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

# Load environment variables from .env
load_dotenv()

# Read config from environment
api_key = os.getenv("IBM_CLOUD_API_KEY")
url = os.getenv("IBM_WATSONX_URL", "https://jp-tok.ml.cloud.ibm.com")
project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
model_id = os.getenv("MODEL_ID", "meta-llama/llama-3-2-11b-vision-instruct")

if not api_key or not project_id:
    raise ValueError(
        "Missing IBM_CLOUD_API_KEY or IBM_WATSONX_PROJECT_ID in environment. "
        "Check your .env file."
    )

# Set credentials to use the model
credentials = Credentials(
    url=url,
    api_key=api_key,
)

# Set necessary parameters (you can customize these later)
params = TextChatParameters()

# Initialize the model
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params,
)

prompt_txt = "How to be a good Data Scientist?"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_txt,
            },
        ],
    }
]

# Generate a response
generated_response = model.chat(messages=messages)
generated_text = generated_response["choices"][0]["message"]["content"]

print(generated_text)
