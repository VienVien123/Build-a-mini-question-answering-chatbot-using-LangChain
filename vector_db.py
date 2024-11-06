from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings, SentenceTransformerEmbeddings
import os
from langchain_huggingface import HuggingFaceEmbeddings


data_path='data'
vector_db_path='vectorstrores/db_faiss' # where to srore db

# creat vector db to 1 text
def creat_db_from_text():
    text="""Nhìn chung, giải quyết một bài toán tối ưu hóa bị ràng buộc là tương đối khó khăn. 
    Có một cách giải quyết bắt nguồn từ vật lý dựa trên một trực giác khá đơn giản. Hãy tưởng tượng
      có một quả banh bên trong một chiếc hộp. Quả banh sẽ lăn đến nơi thấp nhất và trọng lực sẽ cân bằng
        với lực nâng của các cạnh hộp tác động lên quả banh. Tóm lại, gradient của hàm mục tiêu 
        (ở đây là trọng lực) sẽ được bù lại bởi gradient của hàm ràng buộc (cần phải nằm trong chiếc hộp, 
        bị các bức tưởng “đẩy lại”). Lưu ý rằng bất kỳ ràng buộc nào không kích hoạt (quả banh không đụng đến bức tường) 
        thì sẽ không có bất kỳ một lực tác động nào lên quả banh."""
    
    #split text
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size=500, # length per text
        chunk_overlap=30, # length conatence the chacracter of per text
        length_function=len     
    )

    chunks=text_splitter.split_text(text)
    
    # embedding 
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # to faiss vector db
    db=FAISS.from_texts(texts=chunks,embedding=embedding_model)
    # os.makedirs(vector_db_path, exist_ok=True)
    db.save_local(vector_db_path) #save vector db path
    return db
# creat_db_from_text()


# creat vector db to file pdf. 
# if you want to read another file. you can change import laibrary.here my data is pdf
def creat_db_file ():
    loader=DirectoryLoader(data_path,glob='*.pdf',loader_cls=PyPDFLoader)
    # load all document
    documents=loader.load()

    text_spliter=RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlab=50,
    )
    chunks=text_spliter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db=FAISS.from_documents(chunks,embedding_model)

    db.save_local(vector_db_path)
    return db



if __name__ == '__main__': 
    # creat_db_file
     creat_db_from_text()
