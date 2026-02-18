# 🎯 AI Career Coach

AI Career Coach is a set of Python applications that use IBM watsonx.ai and Gradio to provide career-focused assistance, including chat-based guidance, resume polishing, and cover letter generation.

---

## 🚀 Features

- 💬 **LLM Chatbot (`llm_chat.py`)**  
  Ask general career or technical questions and get responses from a large language model.

- 📄 **Resume Polisher (`resume_polisher.py`)**  
  Improve your resume content for a specific position, with optional custom polishing instructions.

- 📨 **Cover Letter Generator (`cover_letter.py`)**  
  Generate customized cover letters tailored to a company, role, and job description or skill keywords.

- 🧭 **Career Advisor (`career_advisor.py`)**  
  Get targeted advice on how to improve your resume to better match a job description.

- 🧪 **Gradio Demo (`gradio_demo.py`)**  
  A simple Gradio example illustrating how the UI framework works.

---

## 🧱 Project Structure

```text
ai-career-coach/
├── my_env/                # Local virtual environment (ignored in git)
├── .env                   # Environment variables (NOT committed)
├── .gitignore
├── requirements.txt       # Python dependencies (recommended)
├── gradio_demo.py
├── llm_chat.py
├── resume_polisher.py
├── cover_letter.py
├── career_advisor.py
└── simple_llm.py
```

> Note: `.env` and `my_env/` are intentionally excluded from version control.

## 🛠️ Prerequisites
* Python 3.11 (or compatible 3.x version)

* A valid IBM Cloud / watsonx.ai account and API key

* Git (if you want to clone and contribute)

## ⚙️ Environment Setup
1. Clone the repository

```bash
git clone https://github.com/aiedwardyi/ai-career-coach.git
cd ai-career-coach
```

2. Create and activate a virtual environment

```bash
python -m venv my_env
# Windows PowerShell
.\my_env\Scripts\Activate.ps1
# or Command Prompt
my_env\Scripts\activate.bat
```

3. Install dependencies

* Using requirements.txt:

```bash
pip install -r requirements.txt
```

* Or install core packages manually:

```bash
pip install gradio==5.12.0 ibm-watsonx-ai==1.1.20 python-dotenv email-validator==2.1.1 numpy==1.26.4 pandas==2.1.4
```

## 🔐 Environment Variables
Create a .env file in the project root (same folder as the .py files) with:

```text
IBM_CLOUD_API_KEY=your_ibm_api_key_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
IBM_WATSONX_PROJECT_ID=your_project_id_here
MODEL_ID=meta-llama/llama-3-2-11b-vision-instruct
```

* Do not commit `.env` to version control.
* These variables are loaded in the scripts using `python-dotenv`.


## ▶️ How to Run
From the project root, with your virtual environment active:

**General chat application**

```bash
python llm_chat.py
```

**Resume polisher**

```bash
python resume_polisher.py
```

**Cover letter generator**

```bash
python cover_letter.py
```

**Career advisor**

```bash
python career_advisor.py
```

**Simple Gradio demo**

```bash
python gradio_demo.py
```

Each script will start a local Gradio web app and print a URL (e.g., http://127.0.0.1:7860) that you can open in your browser.

## 🧪 Testing & Development Notes
* Scripts are structured to read configuration from `.env` using `python-dotenv`.

* Temperature and max_tokens parameters are set per use case:

    * Lower temperature (e.g., 0.1) for more deterministic answers.
    * Higher temperature (e.g., 0.7) for more creative text (e.g., cover letters).

* You can tweak these in the TextChatParameters in each script.

## 📦 Dependencies (summary)
Key libraries used:

* gradio – Web UI for interacting with the models.

* ibm-watsonx-ai – IBM watsonx.ai SDK for accessing foundation models.

* python-dotenv – Loads environment variables from .env.

* numpy, pandas, email-validator – Utility libraries used in parts of the course/project.

## 🤝 Contributing
This project started as part of the “Building Generative AI-Powered Applications with Python” learning path.
Feel free to fork the repository, open issues, or submit pull requests with improvements and new features.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
