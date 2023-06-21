########### GUI IMPORTS ################
import streamlit as st
import ssl
from stqdm import stqdm

############# Displaying images on the front end #################
st.set_page_config(page_title="Rewrite and Paraphraase your Text",
                   page_icon='📖',
                   layout="centered",  #or wide
                   initial_sidebar_state="expanded",
                   menu_items={
                        'Get Help': 'https://docs.streamlit.io/library/api-reference',
                        'Report a bug': "https://www.extremelycoolapp.com/bug",
                        'About': "# This is a header. This is an *extremely* cool app!"
                                },
                   )
########### SSL FOR PROXY ##############
ssl._create_default_https_context = ssl._create_unverified_context

#### IMPORTS FOR AI PIPELINES ###############
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import AutoModel, T5Tokenizer, T5Model
from transformers import T5ForConditionalGeneration
from langchain.llms import HuggingFacePipeline
import torch
import datetime
import os
import requests
# PUT HERE YOUR HUGGING FACE API TOKEN shuold start with hf_XXXXXXX...
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
HUGGING_FACE_API_KEY = "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"

#############################################################################
#               SIMPLE TEXT2TEXT GENERATION INFERENCE
#           checkpoint = "./models/LaMini-Flan-T5-783M.bin" 
# ###########################################################################
checkpoint = "./model/"  #it is actually LaMini-Flan-T5-248M
LaMini = './model/'

######################################################################
#     SUMMARIZATION FROM TEXT STRING WITH HUGGINGFACE PIPELINE       #
######################################################################
def REWRITE(text,chunks, overlap):
    from langchain import HuggingFaceHub

    #The 2 lines below only with API key
    #repo_id = "MBZUAI/LaMini-Flan-T5-248M" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    #llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":512})
    
    llm = HuggingFacePipeline.from_model_id(model_id=LaMini,
                                        task = 'text2text-generation',
                                        model_kwargs={"temperature":0, "max_length":512})
    from langchain import PromptTemplate, LLMChain
    template = """Rewrite the following text mantaining the same length: {text} """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    #It will add punctuation for for you
    # otherwise 500 chunks, 0 overlap
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from tqdm.rich import trange, tqdm
    start = datetime.datetime.now() #not used now but useful
    #chunks = 500
    #overlap = 0
    if len(text) > 500:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = chunks,
            chunk_overlap  = overlap,
            length_function = len,
        )
        texts = text_splitter.split_text(text)
        full_rewrite = ''
        for cnk in stqdm(range(len(texts))):
          result = llm_chain.run(texts[cnk])
          full_rewrite = full_rewrite + ' '+ result
        stop = datetime.datetime.now() #not used now but useful
        delta = stop-start
        return full_rewrite, delta
    else:
        full_rewrite = llm_chain.run(text)
        stop = datetime.datetime.now() #not used now but useful
        delta = stop-start
        return full_rewrite, delta



global full_rewrite

### HEADER section
#st.image('Headline-text.jpg', width=750)
st.title("REWRITE text with AI power")
title = st.text_area('Insert here your Copy/Paste text', "", height = 350, key = 'copypaste')
btt = st.button("1. Start Rewrite process")
txt = st.empty()
timedelta = st.empty()
text_lenght = st.empty()
redux_bar = st.empty()
st.divider()
down_title = st.empty()
down_btn = st.button('2. Download New text') 
text_summary = ''

def start_sum(text):
    if st.session_state.copypaste == "":
        st.warning('You need to paste some text...', icon="⚠️")
    else:
        with st.spinner('Initializing pipelines...'):
            st.success(' AI process started', icon="🤖")
            print("Starting AI pipelines")
            text_rewrite, duration, = REWRITE(text,500,0)
        txt.text_area('Paraphrased text', text_rewrite, height = 350, key='final')
        timedelta.write(f'Completed in {duration}')
        text_lenght.markdown(f"Initial length = {len(text.split(' '))} words / rewrite text = **{len(text_rewrite.split(' '))} words**")
        redux_bar.progress(len(text_rewrite)/len(st.session_state.copypaste), f'Reduction: **{int(1-(len(text_rewrite)/len(st.session_state.copypaste)))*100}** %')
        down_title.markdown(f"## Download your REWRITE text")



if btt:
    start_sum(st.session_state.copypaste)

if down_btn:
    def savefile(generated_summary, filename):
        st.write("Download in progress...")
        with open(filename, 'w') as t:
            t.write(generated_summary)
        t.close()
        st.success(f'AI REWRITE saved in {filename}', icon="✅")
    savefile(st.session_state.final, 'text_REWRITE.txt')
    txt.text_area('Summarized text', st.session_state.final, height = 350)



