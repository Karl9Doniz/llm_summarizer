from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_cohere.llms import Cohere
from typing import Dict

app = FastAPI(
    title="Text Summarization API",
)

class TextInput(BaseModel):
    '''
    Stores the user text input.
    '''

    text: str = Field(..., min_length=1, description="The text to be summarized")

    class Config:
        '''
        Example body of how input request should look like.
        '''

        schema_extra = {
            "example": {
                "text": "Lorem Ipsum is simply dummy text of the printing\
                and typesetting industry. Lorem Ipsum has been the industry's\
                standard dummy text ever since the 1500s, when an unknown\
                printer took a galley of type and scrambled it to make a type\
                specimen book."
            }
        }

class SummaryOutput(BaseModel):
    '''
    Stores the output text processed by the model.
    '''

    summary: str = Field(..., description="The summarized text")

def summarize(text: str) -> str:
    '''
    Using libraries from LangChain framework and Cohere's model, processes
    the text and returns summary.
    '''

    prompt_template = """Write a short summary of the following text: "{text}" """
    prompt = PromptTemplate.from_template(prompt_template)
    llm = Cohere()
    llm_chain = prompt | llm
    summary = llm_chain.invoke({"text": text})
    return summary

@app.post("/summarize", response_model=SummaryOutput)
async def summarize_text(input: TextInput) -> Dict[str, str]:
    '''
    Single POST request function. Handles any unusual error with code 500.
    '''

    try:
        summary = summarize(input.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)