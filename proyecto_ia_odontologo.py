import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from streamlit_navigation_bar import st_navbar
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar las variables de entorno
load_dotenv()

# Obtener la API Key desde las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("La API Key de OpenAI no se ha encontrado. Por favor, verifica tu archivo .env")


# Función para extraer texto de PDFs
def extract_text_from_pdfs(folder_path):
    pdf_texts = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                pdf_texts[file_name] = text
            except Exception as e:
                st.error(f"Error procesando {file_name}: {e}")
    return pdf_texts

# Función para dividir el texto en chunks
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Función para crear un índice FAISS
def create_faiss_index(pdf_texts):
    embeddings = OpenAIEmbeddings()
    texts = []
    metadatas = []
    for file_name, text in pdf_texts.items():
        chunks = split_text_into_chunks(text)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({'document': file_name, 'chunk': i})
    index = FAISS.from_texts(texts, embeddings, metadatas)
    return index

# Función para generar una respuesta basada en la consulta
def generate_response(query, index, top_k=5):
    docs_and_scores = index.similarity_search_with_score(query, k=top_k)
    
    relevant_chunks = ""
    for doc, score in docs_and_scores:
        relevant_chunks += f"\n\nChunk {doc.metadata['chunk']} del documento {doc.metadata['document']}:\n{doc.page_content}"
    
    llm = OpenAI()
    prompt = f"""Soy una inteligencia artificial especializada en odontología. Mi objetivo es ayudarte a encontrar y comprender información basada en una base de datos extensa 
    y actualizada de documentos relevantes en este campo. Puedes hacerme preguntas sobre temas relacionados con la odontología, y yo proporcionaré respuestas detalladas y precisas. 
    Además, cuando te brinde una respuesta, incluiré las referencias más relevantes de los documentos que utilicé para asegurar la transparencia y permitirte consultar la fuente original para obtener más detalles.
    
    Pregunta: {query}
    
    Información relevante:{relevant_chunks}
    
    Respuesta:"""
    response = llm(prompt)
    return response.strip(), relevant_chunks



# Función principal para la aplicación Streamlit
def main():
   
    st.set_page_config(page_title="LUC-IA", page_icon="./img/diente.jpg", layout="wide")

    
    hide_streamlit_style = """
    <style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Aplicar estilos a la barra de navegación usando CSS
    st.markdown(
        """
        <style>
        /* Estilo para los elementos de la barra de navegación */
        .stRadio div {
            background-color: #148EFA; /* Fondo azul */
            color: white; /* Texto en blanco */
            padding: 10px;
            border-radius: 5px;
        }

        /* Cambiar color de las etiquetas de los botones */
        .stRadio label {
            color: white; /* Texto en blanco */
        }

        /* Estilo para la opción seleccionada */
        .stRadio input[type="radio"]:checked + div label {
            background-color: #0d6efd; /* Fondo más oscuro para la opción seleccionada */
            color: white; /* Texto en blanco */
        }
        </style>
        """,
    unsafe_allow_html=True
)

    
    
    # Inicializar variables de estado
    if "index" not in st.session_state:
        st.session_state.index = None

    # Estilo CSS para la app
    st.markdown(
        """
        <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
         
        .css-18e3th9 {
            padding-top: 3px;
        }
        .css-1d391kg {
            background-color: #148EFA;
        }
        .sidebar .sidebar-content {
            background-color: #148EFA;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Aplicar estilo personalizado a la barra lateral
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stHeader"] {
            visibility: hidden;
        }

        [data-testid="stSidebarHeader"] {
            visibility: hidden;
        }

        .stSidebarContent {
            margin-top: 0;
            padding-top: 0;
        }

        .main {
            background-color: #A4D4FE; /* Color de fondo */
        }

        /* Cambiar el color de fondo de la barra lateral */
        .css-1d391kg {
            background-color: #98CCFA; /* Color de fondo */
        }

        /* Estilos para los botones de radio */
        .stRadio {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            margin-top: 10px;
        }

        /* Estilos para los botones */
        .sidebar-buttons {
            display: flex;
            justify-content: space-between; /* Espacio entre los botones */
            padding: 10px;
            background-color: #148EFA; /* Color de fondo para los botones */
            margin-top: auto; /* Empuja los botones hacia abajo */
        }

        .sidebar-button {
            margin: 0; /* Eliminar margen entre los botones */
            padding: 5px 15px;
            color: white;
            background-color: #0056b3; /* Color de fondo de los botones */
            border: none;
            border-radius: 5px;
            cursor: pointer;
            flex-grow: 1; /* Hacer que los botones ocupen el mismo espacio */
            text-align: center; /* Centrar el texto dentro del botón */
        }

        .sidebar-button:hover {
            background-color: #003d7a; /* Color de fondo cuando el cursor está sobre el botón */
        }
            [data-testid="stSidebar"] {
        background-color: #148EFA; /* Color de fondo de la barra lateral */
    }

    .sidebar-buttons {
        display: flex;
        justify-content: space-between; /* Espacio entre los botones */
        padding: 10px;
        background-color: #148EFA; /* Color de fondo para los botones */
        position: fixed;  /* Fijar en la posición */
        bottom: 20px;     /* A 20px de la parte inferior */
        left: 10;
        width: 20%;      /* Asegura que los botones ocupen todo el ancho del sidebar */
        
    }

    .sidebar-button {
        margin: 2px;
        padding: 5px 15px;
        color: white;
        background-color: #FFFFFF; /* Color de fondo de los botones */
        border: none;
        border-radius: 5px;
        cursor: pointer;
        flex-grow: 1; /* Hacer que los botones ocupen el mismo espacio */
        text-align: center;
    }

    .sidebar-button:hover {
        background-color: #003d7a; /* Color de fondo cuando el cursor está sobre el botón */
    }

    /* Estilos para ocultar y mostrar el contenido */
    .tab-content {
        display: none;
        margin-top: 10px;
        color: white;
    }

    #info:target .tab-content,
    #contacto:target .tab-content,
    #referencias:target .tab-content {
        display: block;
    }
    </style>

    
    <div id="info">
        <div class="tab-content">
            <h3>Sobre LUC-IA</h3>
            <p>
                Soy una inteligencia artificial (IA) diseñada para ayudar en la búsqueda y comprensión de información en el campo de la odontología. 
            </p>
            <p>
                Mi base de conocimiento está construida a partir de una amplia gama de documentos y recursos relevantes en odontología, lo que me permite proporcionar respuestas informativas y útiles basadas en el contenido de los PDFs que me proporcionas.
            </p>
            <p>
                Cuando te proporciono una respuesta, también incluyo las referencias más relevantes de la base de conocimiento de la cual se ha obtenido la información. Esto te permite verificar la fuente y profundizar en los detalles si lo deseas.
            </p>
        </div>
    </div>

    <div id="contacto">
        <div class="tab-content">
            <h3>¿Cómo Funcionan las IAs Generativas?</h3>
            <p>
                Las IAs generativas, como yo, utilizan modelos de aprendizaje automático para crear contenido nuevo y coherente basado en datos de entrada. Estos modelos son entrenados con grandes volúmenes de texto para entender el lenguaje humano y generar respuestas contextualmente relevantes. A través de técnicas avanzadas como el procesamiento del lenguaje natural (NLP) y el aprendizaje profundo, las IAs generativas pueden interpretar preguntas, extraer información y formular respuestas en lenguaje natural.
            </p>
        </div>
    </div>

    <div id="referencias">
        <div class="tab-content">
             <h3>Consejos para Sacar el Máximo Provecho</h3>
            <p>
                Para obtener las mejores respuestas de una IA generativa, sigue estos consejos:
            </p>
            <ul>
                <li><b>Proporciona Contexto:</b> Cuanto más contexto proporciones en tu pregunta, más precisa y relevante será la respuesta. Incluye detalles específicos y cualquier información adicional que pueda ayudar a la IA a entender mejor tu consulta.</li>
                <li><b>Utiliza Preguntas Directas:</b> Formule preguntas claras y directas. Evita términos vagos o ambiguos que puedan dificultar la interpretación de tu pregunta.</li>
                <li><b>Revisa y Refina:</b> Si la respuesta no es lo que esperabas, intenta reformular tu pregunta o proporcionar más detalles. La iteración puede mejorar la calidad de las respuestas.</li>
                <li><b>Comprueba la Información:</b> Aunque la IA puede proporcionar información valiosa, siempre es recomendable verificar la exactitud de las respuestas, especialmente en temas críticos o técnicos.</li>
            </ul>
        </div>
    </div>

     <div class="sidebar-buttons">
        <a href="#info" class="sidebar-button">Info</a>
        <a href="#contacto" class="sidebar-button">IA-gen</a>
        <a href="#referencias" class="sidebar-button">Guia</a>
    </div>
    """,
    unsafe_allow_html=True)
 
    
    # Aplicar estilo personalizado a la barra lateral
    st.sidebar.markdown(
        """
        <style>
        /* Cambiar el color de fondo de la barra lateral */
        .css-1d391kg {
            background-color: #98CCFA; /* Color de fondo */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
        

    # Modificar la columna 2
    # Definir columnas
    col1, col2, col3 = st.columns(3)
    with col2:
        # Aplicar color al texto y margen superior de 50px
        st.markdown(
            """
             <style>
                #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            <div style="color: #FF5733; margin-top: -20px; text-align: center;">
                <h1 style="margin-right: 40px; color:white; font-family: fantasy; ">LUC-IA</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Mostrar la imagen a la derecha del texto
        st.image('./img//odontologa.jpg', width=200)
    
    

    folder_path = './documentacion_odontologia'
    
    # Inicializar el estado de sesión si no está definido
    if 'index' not in st.session_state:
        st.session_state.index = None
    
    if st.session_state.index is None:
        with st.spinner("Recopilando información..."):
            pdf_texts = extract_text_from_pdfs(folder_path)
            st.session_state.index = create_faiss_index(pdf_texts)
        st.success("Información recopilada exitosamente")
    
    if st.session_state.index:
        query = st.text_input("Haz una pregunta:")
        if st.button("Generar respuesta"):
            response, relevant_chunks = generate_response(query, st.session_state.index)
            st.write(response)

            st.write("**Referencias**")
            st.write(relevant_chunks)

    
    
if __name__ == "__main__":
    main()
