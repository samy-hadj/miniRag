import os
import faiss  # Install with pip install faiss-cpu
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI  # Importer ChatOpenAI
from dotenv import load_dotenv  # Importer dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)

# Function to call GPT with a given context and question
def ask_gpt_with_context(context, question):
    response = llm.invoke(f"Context: {context}\nQuestion: {question}")
    return response.content  # Adjusted to return just the content of the response

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize FAISS vector store
print("\n=== Initialisation de FAISS ===")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
print("FAISS initialisé avec succès.")
print("======================\n")

# Read documents from data.txt
print("=== Lecture des documents ===")
documents = []
with open('data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(Document(page_content=line.strip(), metadata={"source": "data.txt"}))
print(f"Nombre de documents lus : {len(documents)}")
print("======================\n")

# Generate unique IDs for documents
uuids = [str(uuid4()) for _ in range(len(documents))]

# Add documents to the vector store
print("=== Ajout des documents à FAISS ===")
vector_store.add_documents(documents=documents, ids=uuids)
print("Documents ajoutés à FAISS.")
print("======================\n")

# Tester des questions basées sur le contenu de data.txt
questions = [
    "Qui est le frère de Samy ?",
    "Combien de sœurs Samy a-t-il ?",
    "Où travaille Yanis ?",
    "Quel est le diplôme de Yanis ?",
    "Qui vit chez Samy ?"
]

# Boucle sur chaque question
for question in questions:
    print(f"=== Question : {question} ===")
    
    # Similarity search
    results = vector_store.similarity_search(question, k=2, filter={"source": "data.txt"})
    
    # Affichage des résultats trouvés par FAISS
    context = " ".join([res.page_content for res in results])
    for res in results:
        print(f"Contexte trouvée : {res.page_content} [{res.metadata}]")
    
    # Requête à GPT avec le contexte récupéré
    gpt_response = ask_gpt_with_context(context, question)
    print(f"Réponse de GPT : {gpt_response}")
    
    print("======================\n")
