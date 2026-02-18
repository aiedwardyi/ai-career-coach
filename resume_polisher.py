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

# Function to polish the resume using the model, making polish_prompt optional
def polish_resume(position_name, resume_content, polish_prompt=""):
    # Check if polish_prompt is provided and adjust the combined_prompt accordingly
    if polish_prompt and polish_prompt.strip():
        prompt_use = (
            f"Given the resume content: '{resume_content}', polish it based on the "
            f"following instructions: {polish_prompt} for the {position_name} position."
        )
    else:
        prompt_use = (
            f"Suggest improvements for the following resume content: '{resume_content}' "
            f"to better align with the requirements and expectations of a {position_name} position. "
            f"Return the polished version, highlighting necessary adjustments for clarity, relevance, "
            f"and impact in relation to the targeted role."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_use,
                },
            ],
        }
    ]

    # Generate a response using the model with parameters
    generated_response = model.chat(messages=messages)

    # Extract and return the generated text
    generated_text = generated_response["choices"][0]["message"]["content"]

    return generated_text


# Create Gradio interface for the resume polish application, marking polish_prompt as optional
resume_polish_application = gr.Interface(
    fn=polish_resume,
    flagging_mode="never",  # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(label="Position Name", placeholder="Enter the name of the position..."),
        gr.Textbox(label="Resume Content", placeholder="Paste your resume content here...", lines=20),
        gr.Textbox(
            label="Polish Instruction (Optional)",
            placeholder="Enter specific instructions or areas for improvement (optional)...",
            lines=2,
        ),
    ],
    outputs=gr.Textbox(label="Polished Content"),
    title="Resume Polish Application",
    description=(
        "This application helps you polish your resume. Enter the position you want to apply, "
        "your resume content, and specific instructions or areas for improvement (optional), "
        "then get a polished version of your content."
    ),
)

# Launch the application
if __name__ == "__main__":
    resume_polish_application.launch()
