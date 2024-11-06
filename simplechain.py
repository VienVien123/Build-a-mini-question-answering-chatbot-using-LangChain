# from langchain_community.llms import ctransformers
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# configuration
model_file = 'model/vinallama-7b-chat_q5_0.gguf'

# load llm
def load_llm(model_file):
    llm= CTransformers(
        model=model_file,
        model_type='llama', # type model you choose, here i chose llama
        max_new_tokens=1024, # maximum number of tokens new generated 
        temperature= 0.01 # the level creativity of model. here, i want to model  answered clearly so i choose 0.01
    )

    return llm

# creat prompt
def creat_prompt(template):
    prompt = PromptTemplate (
        template=template,
        input_variables=['question']
        )
    return prompt

# creat simple chain
def creat_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt= prompt, llm=llm)
    return llm_chain


template="""<|im_start|>system
    Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
    <|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant"""


if __name__ == '__main__': 
    #prompt template
    prompt=creat_prompt(template)
    llm=load_llm(model_file)
    llm_chain= creat_simple_chain(prompt, llm)

    # question="hình tam giác nhiêu cạnh"
    # response=llm_chain.invoke({'question': question})
    # print(response)

    


