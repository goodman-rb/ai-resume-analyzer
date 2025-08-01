import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import json
import pandas as pd

# Импортируем наши Pydantic-схемы из файла schema.py
from schema import ResumeData, JobMatchResult

# Загружаем переменные окружения (включая OPENAI_API_KEY)
load_dotenv()

# --- Инициализация состояния сессии ---
# Это нужно, чтобы хранить результаты между перезапусками страницы (например, после нажатия кнопки)
if 'results' not in st.session_state:
    st.session_state.results = None


# --- ЛОГИКА LANGCHAIN ---

@st.cache_data
def process_resume(file_content, file_name):
    """Извлекает структурированные данные из одного файла резюме."""
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    file_path = os.path.join(temp_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return None, "Неподдерживаемый формат файла."

        documents = loader.load()
        resume_text = " ".join([doc.page_content for doc in documents])
        
        parser_resume = PydanticOutputParser(pydantic_object=ResumeData)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        
        prompt_template_resume = """
        Вы — эксперт-рекрутер. Извлеки структурированную информацию из текста резюме.
        Текст резюме: --- {resume_text} ---
        Строго следуй инструкциям по форматированию. {format_instructions}
        """
        prompt_resume = ChatPromptTemplate.from_template(
            template=prompt_template_resume,
            partial_variables={"format_instructions": parser_resume.get_format_instructions()}
        )
        extraction_chain = prompt_resume | llm | parser_resume
        extracted_data = extraction_chain.invoke({"resume_text": resume_text})
        return extracted_data, None
        
    except Exception as e:
        return None, f"Произошла ошибка при извлечении данных из файла {file_name}: {e}"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def run_job_match_analysis(_resume_data_dict, job_description):
    """Анализирует соответствие одного резюме и вакансии."""
    try:
        resume_data_str = json.dumps(_resume_data_dict, ensure_ascii=False, indent=2)
        parser_match = PydanticOutputParser(pydantic_object=JobMatchResult)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

        prompt_template_match = """
        Ты — опытный HR-аналитик. Проанализируй соответствие кандидата требованиям вакансии.
        Вот структурированные данные из резюме кандидата:
        ---
        {resume_data}
        ---
        А вот текст вакансии:
        ---
        {job_description}
        ---
        Твоя задача:
        1. Оценить соответствие по шкале от 1 до 10.
        2. Написать краткое резюме (3-4 ключевых пункта) о том, почему кандидат подходит или не подходит.
        3. Указать, каких ключевых навыков из вакансии не хватает кандидату.
        
        Следуй инструкциям по форматированию. {format_instructions}
        """
        prompt_match = ChatPromptTemplate.from_template(
            template=prompt_template_match,
            partial_variables={"format_instructions": parser_match.get_format_instructions()}
        )
        match_chain = prompt_match | llm | parser_match
        match_result = match_chain.invoke({"resume_data": resume_data_str, "job_description": job_description})
        return match_result, None

    except Exception as e:
        return None, f"Произошла ошибка при анализе соответствия: {e}"

# --- ИНТЕРФЕЙС STREAMLIT ---

st.set_page_config(page_title="AI Resume Analyzer (Batch Mode)", layout="wide")
st.title("🚀 Пакетный анализ резюме")

# --- Блок управления ---
with st.container(border=True):
    st.header("1. Введите данные для анализа")
    job_description_text = st.text_area("Текст вакансии:", height=150, key="job_description")
    uploaded_files = st.file_uploader(
        "Загрузите резюме (можно выбрать несколько файлов)", 
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Начать пакетный анализ", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Пожалуйста, загрузите хотя бы один файл с резюме.")
            elif not job_description_text:
                st.warning("Пожалуйста, вставьте текст вакансии.")
            else:
                all_results = []
                progress_bar = st.progress(0, text="Анализ начат...")

                for i, uploaded_file in enumerate(uploaded_files):
                    progress_text = f"Обрабатываю файл: {uploaded_file.name} ({i+1}/{len(uploaded_files)})..."
                    progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
                    
                    file_bytes = uploaded_file.getvalue()
                    resume_data, error = process_resume(file_bytes, uploaded_file.name)
                    if error:
                        st.error(f"Ошибка в файле {uploaded_file.name}: {error}")
                        continue

                    match_result, match_error = run_job_match_analysis(resume_data.dict(), job_description_text)
                    if match_error:
                        st.error(f"Ошибка анализа для кандидата {resume_data.full_name}: {match_error}")
                        continue

                    all_results.append({
                        "file_name": uploaded_file.name,
                        "resume_data": resume_data,
                        "match_result": match_result
                    })
                
                progress_bar.empty()
                st.session_state.results = all_results # Сохраняем результаты в сессию
    with col2:
        if st.button("Очистить результаты", use_container_width=True):
            st.session_state.results = None
            st.rerun() # Перезапускаем приложение, чтобы очистить интерфейс

# --- Блок отображения результатов ---
if st.session_state.results:
    st.success(f"✅ Анализ завершен! Обработано {len(st.session_state.results)} резюме.")
    
    # --- Создание DataFrame для сводной таблицы ---
    summary_data = []
    for res in st.session_state.results:
        summary_data.append({
            "Имя кандидата": res["resume_data"].full_name,
            "Email": res["resume_data"].email,
            "Оценка": res["match_result"].score,
            "Ключевые выводы": res["match_result"].summary,
            "Недостающие навыки": ", ".join(res["match_result"].missing_skills) if res["match_result"].missing_skills else "Нет"
        })
    df = pd.DataFrame(summary_data)

    # --- Вкладки для отображения ---
    tab_summary, tab_json = st.tabs(["Сводная таблица", "Детальные данные (JSON)"])

    with tab_summary:
        st.header("Сводная таблица результатов")
        st.dataframe(df)

        @st.cache_data
        def convert_df_to_csv(_df):
            return _df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df)
        st.download_button(
            label="📥 Скачать результаты в CSV",
            data=csv,
            file_name="resume_analysis_results.csv",
            mime="text/csv",
        )

    with tab_json:
        st.header("Детальные данные по каждому кандидату")
        for res in st.session_state.results:
            with st.expander(f"📄 {res['resume_data'].full_name or res['file_name']}"):
                st.subheader("Извлеченные данные из резюме")
                st.json(res["resume_data"].model_dump())
                st.subheader("Результаты анализа соответствия")
                st.json(res["match_result"].model_dump())

