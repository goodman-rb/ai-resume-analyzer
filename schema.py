from pydantic import BaseModel, Field
from typing import List, Optional

class Education(BaseModel):
    institution: str = Field(description="Название учебного заведения")
    degree: str = Field(description="Специальность или полученная степень (например, 'Бакалавр', 'Магистр')")
    end_date: Optional[str] = Field(description="Год или дата окончания обучения")

class Experience(BaseModel):
    company: str = Field(description="Название компании")
    position: str = Field(description="Должность")
    start_date: Optional[str] = Field(description="Дата начала работы")
    end_date: Optional[str] = Field(description="Дата окончания работы (может быть 'по настоящее время')")
    description: Optional[str] = Field(description="Краткое описание обязанностей и достижений на этой должности")

class ResumeData(BaseModel):
    full_name: Optional[str] = Field(description="Полное имя и фамилия кандидата")
    email: Optional[str] = Field(description="Контактный адрес электронной почты")
    phone_number: Optional[str] = Field(description="Контактный номер телефона")
    skills: Optional[List[str]] = Field(description="Список ключевых навыков и технологий, которыми владеет кандидат")
    education: Optional[List[Education]] = Field(description="Список полученного образования")
    experience: Optional[List[Experience]] = Field(description="Список мест работы")
    summary: Optional[str] = Field(description="Очень краткая выжимка (1-2 предложения) о профиле кандидата")

# --- НОВЫЙ КЛАСС ДЛЯ РЕЗУЛЬТАТОВ АНАЛИЗА ---
class JobMatchResult(BaseModel):
    score: int = Field(description="Оценка соответствия кандидата вакансии по шкале от 1 до 10")
    summary: str = Field(description="Краткое (3-4 пункта) обоснование оценки, почему кандидат подходит или не подходит")
    missing_skills: Optional[List[str]] = Field(description="Список ключевых навыков из вакансии, которых нет у кандидата")