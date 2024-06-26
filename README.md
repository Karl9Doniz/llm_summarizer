# llm_summarizer

This is a FastAPI application that provides a text summarization service using language models.

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone this repository

2. Create and acivate virtual environment

3. This app uses LangChains built-in Cohere modules,
hence there is a need for external API key.
```
export COHERE_API_KEY=your_api_key_here
```
4. Install needed packages
```
pip install -r requirements.txt
```

5. To run the application, use the following command:
```
uvicorn main:app --reload
```

## Testing

Go to [127.0.0.1/docs](http://127.0.0.1:8000/docs). Try using the UI with API for sending requests.
