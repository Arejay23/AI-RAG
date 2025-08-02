from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load document
loader = TextLoader("./RAG/FAQ_DATA.txt")
documents = loader.load()

# Step 2: Combine documents into a single string (optional, depends on your file)
full_text = "\n".join([doc.page_content for doc in documents])

# Step 3: Split each Q&A block into smaller chunks (if needed)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=520)
chunks = text_splitter.create_documents([full_text])
#total 43 chunks

# Step 4: Create embeddings
embeddings = OpenAIEmbeddings(model ='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 5: Create a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 6: ask question from retriever
question = "what is ?"
retriever_docs=retriever.invoke(question)

context_text = "\n\n".join([doc.page_content for doc in retriever_docs])

# Step 7: create llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""You are a helpful assistant.
    Answer only from provided FAQ context.
    If context is insufficient, just say that you don't know.

    {context}
    {question}
    """,
    input_variables=["context", "question"]
)

# Step 8: ask question from llm.
final_prompt= prompt.invoke({"context":context_text, "question":question})
answer = llm.invoke(final_prompt)
print(answer.content)
print("#"*50)
print(answer)
