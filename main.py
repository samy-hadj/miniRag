import os
import faiss
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configurer la clé API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialiser le modèle de langage
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0)

# Fonction pour interroger GPT avec un contexte donné
def ask_gpt_with_context(context, question):
    response = llm.invoke(f"Context: {context}\nQuestion: {question}")
    return response.content

# Créer les embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialiser le magasin de vecteurs FAISS
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Lire les documents depuis data.txt
documents = []
with open('data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(Document(page_content=line.strip(), metadata={"source": "data.txt"}))

# Générer des IDs uniques pour les documents
uuids = [str(uuid4()) for _ in range(len(documents))]

# Ajouter les documents au magasin de vecteurs
vector_store.add_documents(documents=documents, ids=uuids)

# Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("MiniRag Question Answering System", className="text-center my-4")
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Textarea(
                        id="question-input",
                        placeholder="Enter your question here...",
                        style={"width": "100%", "height": 100},
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Button("Submit", id="submit-button", color="primary", className="mt-4"),
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id="response-output", className="mt-4")
            )
        ),
    ],
    fluid=True,
)

@app.callback(
    Output("response-output", "children"),
    Input("submit-button", "n_clicks"),
    State("question-input", "value"),
)
def update_output(n_clicks, question):
    if n_clicks is None or not question:
        return ""

    # Recherche de similarité
    results = vector_store.similarity_search(question, k=2, filter={"source": "data.txt"})

    # Créer le contexte à partir des résultats de recherche
    context = " ".join([res.page_content for res in results])

    # Obtenir la réponse de GPT
    gpt_response = ask_gpt_with_context(context, question)

    # Afficher le contexte et la réponse
    context_display = html.Div([
        html.H5("Context:"),
        html.P(context),
    ])

    response_display = html.Div([
        html.H5("Response:"),
        html.P(gpt_response),
    ])

    return html.Div([context_display, response_display])

if __name__ == "__main__":
    app.run_server(debug=True)
