from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import LLMChain, PromptTemplate

from server.models import ChatRequest, ChatResponse
from utils.mosaic import MosaicML

load_dotenv()

# Load the language model
llm = MosaicML(max_new_tokens=200).setup()
template = "Person A: {question}\n\nPerson B: "


def chat(query: str) -> str:
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    return llm_chain.run(query)


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    answer = chat(request.query)
    response = ChatResponse(answer=answer)
    return response
