from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
import PyPDF2
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64

load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = my_api_key)

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

@app.post("/")
def ai_prompt(request: ChatRequest):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": request.prompt
            }
        ]
    )

    gpt_response = completion.choices[0].message.content
    return ChatResponse(response = gpt_response)

@app.post("/uploadfile/")
async def create_upload_file(
    prompt: str = Form(...),
    file: UploadFile = File(None)
):
    completion = None
    if file:
        contents = await file.read()
        if file.filename.lower().endswith('.pdf'):
            # Extract text from PDF
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(contents))
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""
            # Use the extracted text as part of the prompt
            full_prompt = f"{prompt}\n\nPDF Content:\n{pdf_text}"
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt}
                ]
            )
        else:
            # Assume image file as before
            base64_image = base64.b64encode(contents).decode("utf-8")
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": prompt },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
            )
    else:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    if completion:
        gpt_response = completion.choices[0].message.content
        return ChatResponse(response = gpt_response)
    return {"message": "No response"}