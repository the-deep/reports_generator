from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel


from summarizer import RepGen


app = FastAPI()
v1 = FastAPI()


class Excerpt(BaseModel):
    language: str = 'en'
    excerpt: Optional[str] = None
    excerpts: Optional[list] = None


@app.get("/")
def home():
    return "Welcome to FastAPI for Reports Generator from DEEP NLP"


@v1.post("/generate_report")
def gen_report(item: Excerpt):
    repgen = RepGen(item.language)

    if item.excerpt:
        input_data = item.excerpt
    elif item.excerpts:
        input_data = item.excerpts
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid request body"
        )

    generated_report = repgen(input_data)
    return {
        "output": generated_report
    }


@v1.post("/uploadfile")
def gen_report_from_file(
    language: str = Form(),
    csv_file: UploadFile = File(
        description="Note: Column names must be id, original_text, groundtruth_labels"
    )
):
    repgen = RepGen(language)

    contents = csv_file.file
    generated_report = repgen.handle_file(contents)

    return {
        "output": generated_report
    }


app.mount("/api/v1", v1)
