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
    max_tokens=512,
)

# Initialize the model
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params,
)

# Function to generate a customized cover letter
def generate_cover_letter(company_name, position_name, job_description, resume_content):
    # Craft the prompt for the model to generate a cover letter
    prompt = (
        f"Generate a customized cover letter using the company name: {company_name}, "
        f"the position applied for: {position_name}, and the job description: {job_description}. "
        f"Ensure the cover letter highlights my qualifications and experience as detailed in the "
        f"resume content: {resume_content}. Adapt the content carefully to avoid including "
        f"experiences not present in my resume but mentioned in the job description. "
        f"The goal is to emphasize the alignment between my existing skills and the "
        f"requirements of the role."
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

    # Extract and return the generated text
    cover_letter = generated_response["choices"][0]["message"]["content"]

    return cover_letter

# Create Gradio interface for the cover letter generation application
cover_letter_app = gr.Interface(
    fn=generate_cover_letter,
    flagging_mode="never",  # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(label="Company Name", placeholder="Enter the name of the company..."),
        gr.Textbox(label="Position Name", placeholder="Enter the name of the position..."),
        gr.Textbox(
            label="Job Skills Keywords",
            placeholder="Paste the job required skills keywords here...",
            lines=2,
        ),
        gr.Textbox(
            label="Resume Content",
            placeholder="Paste your resume content here...",
            lines=10,
        ),
    ],
    outputs=gr.Textbox(label="Customized Cover Letter"),
    title="Customized Cover Letter Generator",
    description=(
        "Generate a customized cover letter by entering the company name, "
        "position name, job description and your resume."
    ),
)

# Launch the application
if __name__ == "__main__":
    cover_letter_app.launch()
