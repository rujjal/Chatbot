#import the necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import ModelScopeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer



def main():

    #cohere api key
    cohere_api_key = "DmNJjqWQ4TIkFaWyC6kS3fc9HEDEjpnkAA2G9c8l"
   # model_id = "damo/nlp_corom_sentence-embedding_english-base"
    st.set_page_config(page_title="CHATBOT for legal queries related to divorce and inheritance ", page_icon=':books:')
    st.header("CHATBOT for legal queries related to divorce and inheritance :books:")

    #memory variable
    memory = []

    # upload file
    pdf = st.file_uploader("Upload your documents", type="pdf", accept_multiple_files=True)
    # extract the text
    texts = []
    if pdf is not None:
        for pdf_file in pdf:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            texts.append(text)

    # split into chunks
    chunks = []
    for text in texts:
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks.extend(text_splitter.split_text(text))

#using the cohere embedding model(open source)
    if st.checkbox("Cohere Embeddings"):
        st.write("Cohere Embeddings Selected!")
        embeddings_cohere = CohereEmbeddings(model= "embed-english-light-v2.0",cohere_api_key=cohere_api_key)
        context = FAISS.from_texts(chunks, embeddings_cohere)

            # show user input
        query = st.text_input("Ask a question about your documents:")
        if query:
            docs = context.similarity_search(query)
            #used cohere llm to answer the queries
            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            # Store the question and answer in memory
            memory.append({"question": query, "answer": response})
           
            st.write(response)
#using model scope embeddings model
    elif st.checkbox("Model Scope Embeddings"):
        model_id = "damo/nlp_corom_sentence-embedding_english-base"
        st.write("Model Scope Embeddings Selected!")
        embeddings_model_scope = ModelScopeEmbeddings(model_id = model_id)
        context = FAISS.from_texts(chunks, embeddings_model_scope)

            # show user input
        query = st.text_input("Ask a question about your documents:")
        if query:
            docs = context.similarity_search(query)

            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
           
            st.write(response)
#using sentence transformers embeddings
    elif st.checkbox("Sentence Transformers Embeddings"):
        st.write("Sentence Transformer Embeddings Selected!")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        embeddings_st = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        context = FAISS.from_texts(chunks, embeddings_st)

            # show user input
        query = st.text_input("Ask a question about your documents:")
        if query:
            docs = context.similarity_search(query)

            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
           
            st.write(response)
#using JINA Ai embedding model
    elif st.checkbox("JINA Ai Embeddings"):
        st.write("JINA Ai Embeddings Selected!")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        embeddings_st = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        context = FAISS.from_texts(chunks, embeddings_st)

            # show user input
        query = st.text_input("Ask a question about your documents:")
        if query:
            docs = context.similarity_search(query)

            llm_cohere = Cohere(model="command", cohere_api_key=cohere_api_key)
            chain = load_qa_chain(llm_cohere, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
           
            st.write(response)

     # Memory section
    if st.checkbox("Memory"):
        st.write("Memory Selected!")

        # Display the memory
        if len(memory) > 0:
            st.write("Previous Questions and Answers:")
            for i, entry in enumerate(memory, 1):
                st.write(f"Q{i}: {entry['question']}")
                st.write(f"A{i}: {entry['answer']}")
        else:
            st.write("No previous questions and answers in memory.")

    # Reset memory
    if st.button("Clear Memory"):
        memory = []
  
if __name__ == '__main__':
    main()