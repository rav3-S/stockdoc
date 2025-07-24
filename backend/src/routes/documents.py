from fastapi import APIRouter, UploadFile, File
from src.utils.pdf_parser import extract_text
from src.utils.summarizer import summarize_report

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text(content)
    summary = summarize_report(text)
    return {"summary": summary}
