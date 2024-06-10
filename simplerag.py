# Import required modules from the LangChain package
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAI
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import chromadb


# Initialize the OpenAI chat model
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)
# llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", temperature=0.8)
llm = ChatOpenAI(base_url="http://localhost:11434/v1",
                 api_key="not-needed",
                 temperature=0.1,
                 model="gemma:7b")

# Initialize the ollama embeddings
embeddings = OllamaEmbeddings(model="gemma:7b")

# Load the Chroma database from disk
# chroma_db = Chroma(persist_directory="data",
#                  embedding_function=embeddings,
#                 collection_name="lc_chroma_demo")

# Get the collection from the Chroma database
# chroma_db = chromadb.HttpClient(host='localhost', port=8000)
client = chromadb.PersistentClient(path="data")

collection = client.get_or_create_collection(name="ls_dagbog")

# collection = chroma_db.get()

# If the collection is empty, create a new one
if collection.count() == 0:
    directory = "filedata"
    # List all PDF files in the directory
    files = [os.path.join(directory, f)
             for f in os.listdir(directory) if f.endswith('.md')]
    # Load a PDF document and split it into sections

    # docs = []  # Initialize an empty list to store the documents

    for file in files:
        # Assuming UnstructuredMarkdownLoader works with a single file at a time
        loader = UnstructuredMarkdownLoader(file, mode="single")
        doc = loader.load_and_split()
        collection.add(ids=[file],
                       documents=doc)
        # docs.extend(doc)  # Add the loaded and split document to the docs list
    print(f"{len(files)} added to DB")

    # print("Length of the list:", len(docs))

    # Example setup of the client to connect to your chroma server

    # chroma_db = chroma_db.create_collection("dagbog", persist_directory="data")

    # chroma_db.add(docs, embedding=embeddings,
    #             persist_directory="data")
    # Create a new Chroma database from the documents
    # chroma_db = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embeddings,
    #     persist_directory="data",
    #     collection_name="ls_dagbog"
    # )

    # Save the Chroma database to disk
    # chroma_db.persist()

print("Retriving docs")
# k is the number of chunks to retrieve
# retriever = chroma_db.as_retriever(k=2)

question = "Hvor mødte jeg Anne?"

retrived_docs = collection.query(query_texts=[question])
# where_document={"$contains": {"text": "search_string"}})

# retrived_docs = retriever.invoke(question)
# print(retrived_docs)

# Example context - this should be replaced with your actual context string

# The question you want to ask

SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""
# Format the SYSTEM_TEMPLATE by inserting the context and preparing the section for the user's question
formatted_system_template = SYSTEM_TEMPLATE.format(context=retrived_docs)

# Combine the formatted system template with the question
full_prompt = f"{formatted_system_template}\nQuestion: {question}\nAnswer:"

print(f"Querying LLM:{full_prompt}")
exit()
# Assume `llm` is already defined as shown in your initial setup
# Directly invoke the LLM to get an answer, adjusting the method call as necessary based on your setup
response = llm.invoke(full_prompt)

# Print the response
print(response)


# question_answering_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             SYSTEM_TEMPLATE,
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

# print("Query LLM")

# chat_res = document_chain.invoke(
#     {
#         "context": retrived_docs,
#         "messages": [
#             HumanMessage(
#                 content="What is there in Flensborg?")
#         ],
#     }
# )

# print(chat_res)
# exit()


# # Prepare query
# query = "What is there in Flensborg?"

# # print('Similarity search:')
# # print(chroma_db.similarity_search(query))

# # print('Similarity search with score:')
# # print(chroma_db.similarity_search_with_score(query))


# chroma_db.similarity_search(query)
# prompt = ChatPromptTemplate.from_template(
#     "Explain thistell me a joke about {foo}")
# model = ChatOpenAI()
# chain = prompt | model


# # Add a custom metadata tag to the first document
# docs[0].metadata = {
#     "tag": "demo",
# }

# # Update the document in the collection
# chroma_db.update_document(
#     document=docs[0],
#     document_id=collection['ids'][0]
# )

# # Find the document with the custom metadata tag
# collection = chroma_db.get(where={"tag": "demo"})


# # Prompt the model
# chain = RetrievalQA.from_chain_type(llm=llm,
#                                     chain_type="stuff",
#                                     retriever=chroma_db.as_retriever())

# # Execute the chain
# response = chain(query)

# # Print the response
# print(response['result'])

# # Delete the collection
# print("Deleing Vector DB")
# chroma_db.delete_collection()
