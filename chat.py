from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Configuration
model_file = 'model/vinallama-7b-chat_q5_0.gguf'
vector_db_path = 'vectorstores/db_faiss'  # where to store db

def create_db_file():
    data_path = 'data'
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding_model)

    db.save_local(vector_db_path)
    return db

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type='llama',
        max_new_tokens=1024,
        temperature=0.01  # lower for clear, precise answers
    )
    return llm

# Create prompt
def create_prompt(template):
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question']
    )
    return prompt

# Create simple chain
def create_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt},
    )
    return llm_chain

# Load vector database
def read_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db


template = """<|im_start|>system
Bạn là một thạc sĩ toán, và bạn đang dạy môn tối ưu hoá. hãy trả lời một cách chính xác và cụ thể cho các câu hỏi. 
<|im_end|>
<|im_start|>user
{context}<|im_end|>
<|im_start|>assistant"""

if __name__ == '__main__':
    if not os.path.exists(vector_db_path):
        print('Không tìm thấy vector db, tạo mới...')
        db = create_db_file()
    else:
        print("Vector db đã tồn tại, tải lên...")
    
    db = read_vector_db()
    print("đọc db thành công")
    
    llm = load_llm(model_file)
    prompt = create_prompt(template)
    llm_chain = create_chain(prompt, llm, db)

    # question = "Giao của các tập lồi có phải tập lôi không?"
    # response = llm_chain.invoke({'query': question})
    # print(response)
    for i in range(4) :
        question = "Giao của các tập lồi có phải tập lôi không?"
        response = llm_chain.invoke({'query': question})
        if i ==3:
            print(response)