import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai as google_genai

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(page_title="RAG arXiv Assistant", page_icon="📚", layout="wide")

# ============================================================
# CARGA DE RECURSOS (se cachea, solo se ejecuta una vez)
# ============================================================

@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

@st.cache_resource
def cargar_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def cargar_coleccion():
    client = chromadb.PersistentClient(path="./arxiv_vector_db")
    return client.get_collection("arxiv_papers")

@st.cache_resource
def cargar_cliente_gemini():
    # Streamlit Community Cloud inyecta los secrets en st.secrets;
    # también soportamos variable de entorno para pruebas locales
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ No se encontró la variable de entorno GEMINI_API_KEY. Configúrala en los Secrets del Space.")
        st.stop()
    return google_genai.Client(api_key=api_key)

model = cargar_modelo_embeddings()
reranker = cargar_reranker()
collection = cargar_coleccion()
gemini_client = cargar_cliente_gemini()

UMBRAL_RELEVANCIA = 0.35

# ============================================================
# FUNCIONES DEL PIPELINE RAG (D, E, F)
# ============================================================

def buscar(query, top_k=20):
    query_embedding = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

    resultados = []
    for meta, documento, distancia in zip(
        results["metadatas"][0], results["documents"][0], results["distances"][0]
    ):
        resultados.append({
            "title": meta["title"],
            "terms": meta["terms"],
            "summary": meta["summary"],
            "texto": documento,
            "distance": distancia
        })
    return resultados


def buscar_con_reranking(query, top_k_inicial=20, top_k_final=5):
    candidatos = buscar(query, top_k=top_k_inicial)
    pares = [(query, c["texto"]) for c in candidatos]
    scores = reranker.predict(pares)

    for c, score in zip(candidatos, scores):
        c["rerank_score"] = float(score)

    candidatos_ordenados = sorted(candidatos, key=lambda x: x["rerank_score"], reverse=True)
    return candidatos_ordenados[:top_k_final]


def generar_respuesta_rag(query, top_k=5, umbral_relevancia=UMBRAL_RELEVANCIA):
    evidencias = buscar_con_reranking(query, top_k_inicial=20, top_k_final=top_k)
    mejor_score = max(e["rerank_score"] for e in evidencias)

    if mejor_score < umbral_relevancia:
        return {
            "respuesta": "El corpus no contiene información suficiente para responder esta consulta con confianza.",
            "evidencias": evidencias,
            "suficiente": False
        }

    contexto = "\n\n".join(
        f"[Documento {i+1}] Título: {e['title']}\nTopics: {e['terms']}\nResumen: {e['summary']}"
        for i, e in enumerate(evidencias)
    )

    prompt = f"""Eres un asistente experto en literatura científica. Responde la siguiente
pregunta del usuario basándote ÚNICAMENTE en los documentos de contexto proporcionados.

Si el contexto no contiene información suficiente para responder, indícalo explícitamente
en vez de inventar información. Responde en el mismo idioma de la pregunta del usuario.

Contexto:
{contexto}

Pregunta: {query}

Respuesta (cita los documentos relevantes usando [Documento N]):"""

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return {
        "respuesta": response.text,
        "evidencias": evidencias,
        "suficiente": True
    }

# ============================================================
# INTERFAZ CONVERSACIONAL (G)
# ============================================================

st.title("📚 Asistente RAG sobre papers de arXiv")
st.caption("Consultas en lenguaje natural sobre un corpus de abstracts científicos (arXiv Paper Abstracts)")

if "historial" not in st.session_state:
    st.session_state.historial = []  # cada item: {"query":..., "respuesta":..., "evidencias":..., "suficiente":...}

# Mostramos el historial de la sesión
for turno in st.session_state.historial:
    with st.chat_message("user"):
        st.write(turno["query"])
    with st.chat_message("assistant"):
        st.write(turno["respuesta"])
        with st.expander("📄 Ver evidencias utilizadas"):
            if not turno["suficiente"]:
                st.info("No se encontró evidencia suficientemente relevante en el corpus.")
            for i, e in enumerate(turno["evidencias"], 1):
                st.markdown(f"**[Documento {i}]** (score de relevancia: `{e['rerank_score']:.4f}`)")
                st.markdown(f"*Título:* {e['title']}")
                st.markdown(f"*Topics:* {e['terms']}")
                st.markdown(f"*Resumen:* {e['summary'][:400]}...")
                st.markdown("---")

# Entrada de nueva consulta
query = st.chat_input("Escribe tu consulta sobre el corpus de arXiv...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el corpus y generando respuesta..."):
            resultado = generar_respuesta_rag(query)
        st.write(resultado["respuesta"])
        with st.expander("📄 Ver evidencias utilizadas"):
            if not resultado["suficiente"]:
                st.info("No se encontró evidencia suficientemente relevante en el corpus.")
            for i, e in enumerate(resultado["evidencias"], 1):
                st.markdown(f"**[Documento {i}]** (score de relevancia: `{e['rerank_score']:.4f}`)")
                st.markdown(f"*Título:* {e['title']}")
                st.markdown(f"*Topics:* {e['terms']}")
                st.markdown(f"*Resumen:* {e['summary'][:400]}...")
                st.markdown("---")

    st.session_state.historial.append({
        "query": query,
        "respuesta": resultado["respuesta"],
        "evidencias": resultado["evidencias"],
        "suficiente": resultado["suficiente"]
    })
