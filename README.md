# Content Evaluation Project

Electrolux creative performance portal built with Streamlit. The app measures uploaded image quality with Python metrics, sends the image to OpenAI for 1-10 creative scoring, creates an AI conclusion with improvement suggestions, and can generate an improved image from the original creative plus all scoring inputs.

## Credentials

Credentials are not committed to the repository or baked into the Docker image.

Create a local `.env` file from the template and add your key:

```bash
cp .env.example .env
```

Required values:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.5
OPENAI_IMAGE_MODEL=gpt-image-1.5
```

`OPENAI_MODEL` controls the scoring and conclusion model. `OPENAI_IMAGE_MODEL` controls the image generation model shown in the app setup panel.

## Run With Docker

```bash
docker compose up --build
```

Open the app at:

```text
http://localhost:8501
```

## Run Locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Project Files

- `app.py`: Streamlit web app and UI
- `ai_analysis.py`: OpenAI scoring prompts, schema control, AI conclusion logic, and improved image generation
- `image_metrics.py`: Python image quality metrics and scoring
- `image/`: Electrolux logo and sample media assets