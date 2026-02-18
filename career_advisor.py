import os

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
if not IBM_WATSONX_URL:
    raise ValueError("IBM_WATSONX_URL is not set. Check your .env file.")
if not IBM_WATSONX_PROJECT_ID:
    raise ValueError("IBM_WATSONX_PROJECT_ID is not set. Check your .env file.")

# 4. Set credentials to use the model (now using env values)
credentials = Credentials(
    url=IBM_WATSONX_URL,
    api_key=IBM_CLOUD_API_KEY,
)

# 5. Model and project settings
model_id = MODEL_ID
project_id = IBM_WATSONX_PROJECT_ID

# Generation parameters
params = TextChatParameters(
    temperature=0.7,
    max_tokens=1024,
)

# Initialize the model
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params,
)

# Function to generate career advice
def generate_career_advice(position_applied, job_description, resume_content):
    # The prompt for the model
    prompt = (
        f"Considering the job description: {job_description}, and the resume provided: {resume_content}, "
        f"identify areas for enhancement in the resume. Offer specific suggestions on how to improve these "
        f"aspects to better match the job requirements and increase the likelihood of being selected for "
        f"the position of {position_applied}."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # Generate a response using the model with parameters
    generated_response = model.chat(messages=messages)

    # Extract and format the generated text
    advice = generated_response["choices"][0]["message"]["content"]
    return advice

# Create Gradio interface for the career advice application
career_advice_app = gr.Interface(
    fn=generate_career_advice,
    flagging_mode="never",  # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(
            label="Position Applied For",
            placeholder="Enter the position you are applying for...",
        ),
        gr.Textbox(
            label="Job Description Information",
            placeholder="Paste the job description here...",
            lines=10,
        ),
        gr.Textbox(
            label="Your Resume Content",
            placeholder="Paste your resume content here...",
            lines=10,
        ),
    ],
    outputs=gr.Textbox(label="Advice"),
    title="Career Advisor",
    description=(
        "Enter the position you're applying for, paste the job description, "
        "and your resume content to get advice on what to improve for getting this job."
    ),
)

# Launch the application
if __name__ == "__main__":
    career_advice_app.launch()
