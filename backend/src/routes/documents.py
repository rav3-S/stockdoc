from fastapi import APIRouter, UploadFile, File
from src.utils.pdf_parser import extract_text
from src.utils.summarizer import summarize_report
from src.utils.vectorstore import create_vectorstore_from_text
from src.utils.qa_engine import get_qa_chain

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text(content)
    summary = summarize_report(text)
    return {"summary": summary}

@router.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = ""):
    content = await file.read()
    text = extract_text(content)
    vectorstore = create_vectorstore_from_text(text)
    qa = get_qa_chain(vectorstore)
    result = qa.run(question)
    return {"answer": result}