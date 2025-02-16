# Multi-Mode Chat Assistant

Это приложение представляет собой многорежимного чат-ассистента, разработанного с использованием Streamlit и Llama Index. Приложение использует Qdrant для работы с векторной базой данных и Ollama для генерации ответов с языковой модели.

## Особенности

- **Режим RAG (Retrieval-Augmented Generation):** Поиск и генерация ответов на основе локальной базы знаний (PDF-документы).
- **Обычный чат:** Стандартный режим общения без интеграции базы знаний.
- Управление историей чатов: создание, переименование, удаление чатов и добавление системных сообщений.

## Требования

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Llama Index](https://github.com/jerryjliu/llama_index)
- [Qdrant Client](https://github.com/qdrant/qdrant_client)
- Локально запущенный сервер [Qdrant](https://qdrant.tech/) (по умолчанию: `localhost:6333`)
- [Ollama](https://ollama.ai/) – установите и настройте согласно документации Ollama

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/KhudV/local_llm_app.git
   cd <название-папки>
   ```
2. **Создайте виртуальное окружение и активируйте его:**

    ```bash
    python -m venv venv
    # На Linux/Mac:
    source venv/bin/activate
    # На Windows:
    venv\Scripts\activate
    ```
3. **Установите зависимости:**

    ```bash
    pip install -r requirements.txt
    ```
## Конфигурация
В файле приложения измените переменную input_dir_path, чтобы указать путь к вашей базе знаний (директория с PDF-документами).
Убедитесь, что сервер Qdrant запущен на localhost:6333 или скорректируйте настройки подключения.
Приложение использует Ollama для генерации ответов. Убедитесь, что Ollama установлен и настроен.
## Запуск приложения
1. Выполните файл local-deepseek.bat

2. Или запустите приложение командой:
   ```bash
    streamlit run local-deepseek.py.py
   ```
## Лицензия
Этот проект распространяется под лицензией MIT.
