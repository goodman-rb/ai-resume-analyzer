AI Resume Analyzer 🤖
<p align="center">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python version">
<img src="https://img.shields.io/badge/Streamlit-1.47-orange?style=for-the-badge&logo=streamlit" alt="Streamlit version">
<img src="https://img.shields.io/badge/LangChain-0.3-green?style=for-the-badge" alt="LangChain version">
<img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-black?style=for-the-badge&logo=openai" alt="OpenAI model">
</p>

<p align="center">
<i>Интеллектуальный инструмент для пакетного анализа резюме и их оценки на соответствие вакансии.</i>
</p>

<p align="center">
<img src="https://i.imgur.com/your-gif-url.gif" alt="Демонстрация работы приложения">
</p>

AI Resume Analyzer — это веб-приложение, созданное для кардинального ускорения процесса рекрутинга. Оно позволяет HR-специалистам загружать десятки резюме, автоматически извлекать из них ключевую информацию и, самое главное, оценивать соответствие каждого кандидата открытой вакансии.

🚀 Ключевые возможности
Пакетная обработка: Загружайте и анализируйте несколько резюме одновременно (в форматах .pdf и .docx).

Интеллектуальный анализ: Приложение не просто извлекает данные, а анализирует их в контексте предоставленной вакансии.

Оценка кандидатов: Автоматически выставляет оценку соответствия каждого кандидата по 10-балльной шкале.

Сводная таблица: Все результаты представляются в виде удобной таблицы для быстрого сравнения кандидатов.

Детальный разбор: Предоставляет краткие выводы по каждому кандидату и список недостающих навыков.

Экспорт данных: Позволяет скачать сводную таблицу в формате .csv для дальнейшей работы в Excel или Google Sheets.

🛠️ Технологический стек
Язык: Python

AI/LLM: LangChain (для оркестрации), OpenAI API (gpt-4o-mini)

Веб-интерфейс: Streamlit

Валидация данных: Pydantic (для надежного извлечения структурированных данных)

Обработка данных: Pandas

⚙️ Как запустить локально
Следуйте этим шагам, чтобы запустить проект на вашем компьютере.

1. Клонируйте репозиторий:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

2. Создайте и активируйте виртуальное окружение:

# Для macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Для Windows
python -m venv venv
.\venv\Scripts\activate

3. Установите зависимости:

pip install -r requirements.txt

4. Настройте API ключ:

Создайте файл с именем .env в корневой папке проекта.

Добавьте в него ваш API ключ от OpenAI:

OPENAI_API_KEY="sk-..."

5. Запустите приложение:

streamlit run app.py

Приложение должно открыться в вашем браузере по адресу http://localhost:8501.
