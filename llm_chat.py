import os

# New import to load .env
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr

# 1. Load environment variables from .env
load_dotenv()

# 2. Read values from environment instead of hardcoding
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_WATSONX_URL = os.getenv("IBM_WATSONX_URL")
IBM_WATSONX_PROJECT_ID = os.getenv("IBM_WATSONX_PROJECT_ID")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/llama-3-2-11b-vision-instruct")

# 3. Basic validation so you notice missing values early
if not IBM_CLOUD_API_KEY:
    raise ValueError("IBM_CLOUD_API_KEY is not set. Check your .env file.")
if not IBM_WATSONX_PROJECT_ID:
    raise ValueError("IBM_WATSONX_PROJECT_ID is not set. Check your .env file.")

# 4. Set credentials to use the model (now using env values)
credentials = Credentials(
    url=IBM_WATSONX_URL,
    api_key=IBM_CLOUD_API_KEY,
)

# 5. Model and project settings (using env-driven MODEL_ID / PROJECT_ID)
model_id = MODEL_ID
project_id = IBM_WATSONX_PROJECT_ID

params = TextChatParameters(
    temperature=0.1,
    max_tokens=512,
)

# Initialize the model
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params,
)

# Function to generate a response from the model
def generate_response(prompt_txt: str) -> str:
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
    generated_response = model.chat(messages=messages)
    generated_text = generated_response["choices"][0]["message"]["content"]
    return generated_text

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer.",
)

# Launch the app
if __name__ == "__main__":
    chat_application.launch()
