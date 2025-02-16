import streamlit as st
from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import re
import json
from pathlib import Path

# ======= Конфигурация ========
input_dir_path = "C:\RAG\Knowledge base"
collection_name = "chat_with_docs"
# Удаляем модель для HuggingFace API – остаются только модели для Ollama
available_models = {
    "deepseek-r1:14b": "deepseek-r1:14b",
    "deepseek-r1-abliterated:14b": "huihui_ai/deepseek-r1-abliterated:14b"
}
HISTORY_FILE = Path("chat_history.json")

# ======= Функции работы с историей ========
def load_chat_history():
    """Загрузка истории чатов из файла"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Конвертация старого формата
                converted_chats = {}
                for chat_name, content in data["chats"].items():
                    if isinstance(content, list):  # Старый формат
                        converted_chats[chat_name] = {
                            "messages": content,
                            "system_messages": []
                        }
                    else:  # Новый формат
                        converted_chats[chat_name] = content
                
                st.session_state.chats = converted_chats
                st.session_state.current_chat = data["current_chat"]
                
    except Exception as e:
        st.error(f"Ошибка загрузки истории: {str(e)}")
        st.session_state.chats = {
            "Чат 1": {
                "messages": [],
                "system_messages": []
            }
        }
        st.session_state.current_chat = "Чат 1"

def save_chat_history():
    """Сохранение истории чатов в файл"""
    try:
        data = {
            "chats": st.session_state.chats,
            "current_chat": st.session_state.current_chat
        }
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Ошибка сохранения истории: {str(e)}")

# ======= Инициализация компонентов ========
if "model_name" not in st.session_state:
    st.session_state.model_name = "deepseek-r1:14b"
if "temperature" not in st.session_state:  # Добавляем температуру
    st.session_state.temperature = 0.6  # 0.6 - рекомендация разработчиков deepseek-r1

@st.cache_resource(show_spinner=False)
def initialize_rag_components(model_name, temperature):
    try:
        # Только логика без Streamlit элементов
        loader = SimpleDirectoryReader(
            input_dir=input_dir_path,
            required_exts=[".pdf"],
            recursive=True
        )
        docs = loader.load_data()
        
        if len(docs) == 0:
            raise ValueError("В директории нет документов!")

        embed_model = HuggingFaceEmbedding(
            model_name="Snowflake/snowflake-arctic-embed-m",
            trust_remote_code=True
        )

        client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            docs,
            embed_model=embed_model,
            storage_context=storage_context
        )

        query_engine = index.as_query_engine(
            similarity_top_k=10, 
            llm=initialize_llm(model_name, temperature),
            streaming=True,  # Включаем потоковый режим
            response_mode="compact"  # Или "tree_summarize" для лучшей потоковой обработки
        )
        
        qa_prompt_tmpl_str = """
        You are an AI assistant specialized in answering user queries based on the provided context. 
        If the answer is within the given context, provide a concise and accurate response.
        If the answer is not in the context, state that you don't know instead of making up information.

        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:
        """

        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        
        return {
            "query_engine": query_engine,
            "docs_count": len(docs),
            "embed_model": "Snowflake/snowflake-arctic-embed-m",
            "llm_model": model_name
        }
            
    except Exception as e:
        raise RuntimeError(f"Ошибка инициализации RAG: {str(e)}")

@st.cache_resource(show_spinner=False)
def initialize_llm(model_name, temperature):
    try:
        # Теперь всегда используем Ollama
        return Ollama(
            model=model_name,
            temperature=temperature,
            request_timeout=300.0
        )
    except Exception as e:
        raise e

# ======= Инициализация приложения ========
st.set_page_config(
    page_title="Multi-Mode Chat",
    page_icon="🤖",
    layout="centered"
)

# Инициализация состояния сессии
if "chats" not in st.session_state:
    load_chat_history()  # Загружаем историю при первом запуске

# Кастомные стили
st.markdown("""
    <style>
    .stTextArea textarea {
        min-height: 100px !important;
        max-height: 200px !important;
    }
    .chat-list {
        max-height: 70vh;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Инициализация состояния сессии
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Чат 1": {"messages": [], "system_messages": []}
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Чат 1"

if "processing" not in st.session_state:
    st.session_state.processing = False

if "mode" not in st.session_state:
    st.session_state.mode = "Обычный чат"

# ======= Боковая панель с настройками ========
with st.sidebar:
    st.title("⚡ История чатов")
    
    if st.button("🆕 Новый чат"):
        new_chat_number = len(st.session_state.chats) + 1
        new_chat_name = f"Чат {new_chat_number}"
        while new_chat_name in st.session_state.chats:
            new_chat_number += 1
            new_chat_name = f"Чат {new_chat_number}"
        st.session_state.chats[new_chat_name] = {"messages": [], "system_messages": []}
        st.session_state.current_chat = new_chat_name
        save_chat_history()
        st.rerun()
    
    st.markdown('<div class="chat-list">', unsafe_allow_html=True)
    for chat_name in list(st.session_state.chats.keys()):
        if st.session_state.get("editing_chat") == chat_name:
            new_name = st.text_input(
                "Новое название:",
                value=chat_name,
                key=f"edit_{chat_name}",
                label_visibility="collapsed",
                on_change=lambda: st.session_state.update({"text_input_edited": True})
            )
            
            if "text_input_edited" in st.session_state:
                if new_name and new_name != chat_name:
                    if new_name not in st.session_state.chats:
                        st.session_state.chats[new_name] = st.session_state.chats.pop(chat_name)
                        if chat_name == st.session_state.current_chat:
                            st.session_state.current_chat = new_name
                        del st.session_state.editing_chat
                        del st.session_state.text_input_edited
                        save_chat_history()
                        st.rerun()
                    else:
                        st.error("Название уже существует!")
                del st.session_state.text_input_edited
            
            cols = st.columns([6, 5, 1])
            with cols[0]:
                if st.button("✅ Сохранить", key=f"save_{chat_name}"):
                    if new_name and new_name != chat_name:
                        if new_name not in st.session_state.chats:
                            st.session_state.chats[new_name] = st.session_state.chats.pop(chat_name)
                            if chat_name == st.session_state.current_chat:
                                st.session_state.current_chat = new_name
                            del st.session_state.editing_chat
                            st.rerun()
                        else:
                            st.error("Название уже существует!")
            with cols[1]:
                if st.button("❌ Отмена", key=f"cancel_{chat_name}"):
                    del st.session_state.editing_chat
                    if "text_input_edited" in st.session_state:
                        del st.session_state.text_input_edited
                    st.rerun()
        
        else:
            cols = st.columns([9, 2, 2])
            with cols[0]:
                is_active = chat_name == st.session_state.current_chat
                btn_type = "primary" if is_active else "secondary"
                
                if st.button(
                    chat_name,
                    key=f"btn_{chat_name}",
                    use_container_width=True,
                    type=btn_type
                ):
                    if is_active:
                        st.session_state.editing_chat = chat_name
                    else:
                        st.session_state.current_chat = chat_name
                    st.rerun()
            
            with cols[1]:
                if st.button("✏️", key=f"edit_btn_{chat_name}", help="Переименовать чат"):
                    st.session_state.editing_chat = chat_name
                    st.rerun()
            
            with cols[2]:
                if st.button("🗑️", key=f"del_{chat_name}", help="Удалить чат"):
                    if len(st.session_state.chats) > 1:
                        del st.session_state.chats[chat_name]
                        if chat_name == st.session_state.current_chat:
                            st.session_state.current_chat = next(iter(st.session_state.chats))
                        save_chat_history()
                        st.rerun()
                    else:
                        st.toast("Нельзя удалить последний чат!", icon="❌")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Настройки
    st.title("⚙️ Настройки")
    # Виджет выбора модели – опция HuggingFace удалена
    current_model_key = [k for k, v in available_models.items() if v == st.session_state.model_name][0]
    new_model_key = st.selectbox(
        "Выберите модель:",
        options=list(available_models.keys()),
        index=list(available_models.keys()).index(current_model_key),
        key="model_selector"
    )
    if available_models[new_model_key] != st.session_state.model_name:
        st.session_state.model_name = available_models[new_model_key]
        st.toast(f"Модель изменена на: {new_model_key}", icon="ℹ️")
        initialize_rag_components.clear()
        initialize_llm.clear()
        st.session_state.llm = initialize_llm(st.session_state.model_name, st.session_state.temperature)
        st.rerun()
    
    new_temp = st.slider(
        "Температура генерации:",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Чем выше значение — тем креативнее ответы. Ниже — более точные и детерминированные."
    )
    
    if new_temp != st.session_state.temperature:
        st.session_state.temperature = new_temp
        st.toast(f"Температура изменена на: {new_temp}", icon="🌡️")
        initialize_llm.clear()
        st.session_state.llm = initialize_llm(st.session_state.model_name, st.session_state.temperature)
        st.rerun()

    new_mode = st.radio(
        "Режим работы:",
        ["RAG", "Обычный чат"],
        index=0 if st.session_state.mode == "RAG" else 1
    )
    
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.chats[st.session_state.current_chat]["messages"] = []
        st.rerun()

    if st.session_state.mode == "RAG":
        try:
            if "rag_info" not in st.session_state:
                with st.status("🔄 Инициализация RAG системы...") as status:
                    st.write("📂 Загрузка документов...")
                    st.session_state.rag_info = initialize_rag_components(st.session_state.model_name, st.session_state.temperature)
                    status.update(label="✅ RAG система готова", state="complete")
                    st.toast("RAG система успешно инициализирована!", icon="🎉")
            
            st.subheader("Параметры RAG")
            st.write(f"**Модель эмбеддингов:**\n{st.session_state.rag_info['embed_model']}")
            st.write(f"**Документов в базе:**\n{st.session_state.rag_info['docs_count']}")
            st.write(f"**Языковая модель:**\n{st.session_state.rag_info['llm_model']}")
            st.write(f"**Температура языковой модели:** {st.session_state.temperature}")
            st.write(f"**Векторная БД:**\nQdrant ({collection_name})")
            
        except Exception as e:
            st.error(str(e))
            st.stop()
    else:
        try:
            if "llm" not in st.session_state:
                with st.spinner(f"Загрузка {st.session_state.model_name}..."):
                    st.session_state.llm = initialize_llm(st.session_state.model_name, st.session_state.temperature)
                    st.toast(f"✅ {st.session_state.model_name} загружена", icon="🎉")
            
            st.subheader("Параметры чата")
            st.write(f"**Режим:** Стандартный чат")
            st.write(f"**Модель:** {st.session_state.model_name}")
            st.write(f"**Температура:** {st.session_state.temperature}")
            st.warning("Используется без базы знаний")
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {str(e)}")
            st.stop()

    st.title("⚙️ Системные сообщения чата")
    
    current_system_messages = st.session_state.chats[st.session_state.current_chat]["system_messages"]
    
    for idx, sm in enumerate(current_system_messages):
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            st.info(sm)
        with cols[1]:
            if st.button("🗑️", key=f"del_sm_{idx}"):
                st.session_state.chats[st.session_state.current_chat]["system_messages"].pop(idx)
                save_chat_history()
                st.rerun()
    
    if st.button("➕ Добавить системное сообщение"):
        st.session_state.adding_sm = True
    
    if st.session_state.get("adding_sm"):
        with st.form(key="add_sm_form"):
            new_sm = st.text_area("Системное сообщение для текущего чата:")
            cols = st.columns(2)
            with cols[0]:
                if st.form_submit_button("Добавить"):
                    if new_sm.strip():
                        st.session_state.chats[st.session_state.current_chat]["system_messages"].append(new_sm.strip())
                        save_chat_history()
                        st.session_state.adding_sm = False
                        st.rerun()
            with cols[1]:
                if st.form_submit_button("Отмена"):
                    st.session_state.adding_sm = False
                    st.rerun()

# ======= Основной интерфейс ========
st.title("💬 Multi-Mode Chat Assistant")

chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.chats[st.session_state.current_chat]["messages"]):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            if st.session_state.get("editing_index") == i and msg["role"] == "user":
                with st.form(key=f"edit_form_{i}"):
                    edited_msg = st.text_area(
                        "Редактировать сообщение",
                        value=msg["content"],
                        key=f"edit_input_{i}",
                        height=100
                    )
                    
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        if st.form_submit_button("✅ Сохранить"):
                            st.session_state.chats[st.session_state.current_chat]["messages"][i]["content"] = edited_msg
                            st.session_state.chats[st.session_state.current_chat]["messages"] = st.session_state.chats[st.session_state.current_chat]["messages"][:i+1]
                            del st.session_state.editing_index
                            st.session_state.processing = True
                            save_chat_history()
                            st.rerun()
                    with cols[1]:
                        if st.form_submit_button("❌ Отмена"):
                            del st.session_state.editing_index
                            st.rerun()
                    with cols[2]:
                        if st.form_submit_button("🗑️ Удалить"):
                            del st.session_state.chats[st.session_state.current_chat]["messages"][i]
                            del st.session_state.editing_index
                            st.rerun()
            
            else:
                cols = st.columns([8, 1])
                with cols[0]:
                    text = msg["content"]
                    parts = re.split(r'(<think>.*?</think>)', text, flags=re.DOTALL)
                    for part in parts:
                        if part.startswith('<think>') and '</think>' in part:
                            with st.expander("🤔 Мыслительный процесс", expanded=False):
                                st.markdown(part)
                        else:
                            st.markdown(part)
                
                if msg["role"] == "user":
                    with cols[1]:
                        if st.button("✏️", key=f"edit_{i}", help="Редактировать ответ"):
                            st.session_state.editing_index = i
                            st.rerun()
                elif msg["role"] == "assistant":
                    with cols[1]:
                        if st.button("🔄", key=f"regenerate_{i}", help="Перегенерировать ответ"):
                            current_chat = st.session_state.chats[st.session_state.current_chat]["messages"]
                            st.session_state.chats[st.session_state.current_chat]["messages"] = current_chat[:i]
                            st.session_state.processing = True
                            save_chat_history()
                            st.rerun()

if st.button("🧹 Очистить историю"):
    st.session_state.chats[st.session_state.current_chat]["messages"] = []
    st.session_state.processing = False
    save_chat_history()
    st.rerun()

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Ваш вопрос:" if st.session_state.mode == "RAG" else "Ваше сообщение:",
        placeholder="Введите ваш текст... (Shift+Enter для новой строки)",
        key="input",
        height=100
    )
    submit_button = st.form_submit_button("Отправить")

if submit_button and user_input:
    if user_input.startswith("*sysmsg*"):
        cleaned_input = user_input.replace("*sysmsg*", "", 1).strip()
        st.session_state.chats[st.session_state.current_chat]["system_messages"].append(cleaned_input)
        save_chat_history()
        st.rerun()
    else:
        st.session_state.chats[st.session_state.current_chat]["messages"].append({
            "role": "user", 
            "content": user_input
        })
        st.session_state.processing = True
        save_chat_history()
        st.rerun()

def prepare_context(messages, system_messages):
    """Добавляет системные сообщения к последнему запросу пользователя"""
    if not system_messages:
        return messages
    
    processed_messages = [msg.copy() for msg in messages]
    
    last_user_msg = None
    for msg in reversed(processed_messages):
        if msg["role"] == "user":
            last_user_msg = msg
            break
    
    if last_user_msg:
        system_block = "## System instructions:\n" + "\n".join(
            f"- {sm}" for sm in system_messages
        )
        last_user_msg["content"] = f"{system_block}\n## User query:\n{last_user_msg['content']}"
    
    return processed_messages

def process_response_chunks(stream_generator):
    """Обрабатывает потоковые чанки и возвращает полный ответ без отложенного отображения блока <think>"""
    buffer = ""
    response_container = st.empty()
    spinner_placeholder = st.empty()

    with spinner_placeholder:
        st.spinner("🤔 Размышление...")

    try:
        if st.session_state.mode == "RAG":
            for char in stream_generator:
                buffer += char
                spinner_placeholder.empty()
                response_container.markdown(buffer + "▌")
        else:
            for chunk in stream_generator:
                delta = chunk.delta if hasattr(chunk, 'delta') else chunk
                buffer += delta
                spinner_placeholder.empty()
                response_container.markdown(buffer + "▌")
    finally:
        spinner_placeholder.empty()

    response_container.empty()
    parts = re.split(r'(<think>.*?</think>)', buffer, flags=re.DOTALL)
    with response_container:
        for part in parts:
            if part.strip():
                if part.startswith('<think>') and '</think>' in part:
                    with st.expander("🤔 Мыслительный процесс", expanded=False):
                        st.markdown(part)
                else:
                    st.markdown(part)
    
    return buffer

if st.session_state.processing:
    with st.spinner("🔍 Поиск информации..." if st.session_state.mode == "RAG" else "💭 Генерирую ответ..."):
        try:
            last_message = st.session_state.chats[st.session_state.current_chat]["messages"][-1]
            if last_message["role"] == "user":
                user_input = last_message["content"]
                
                if st.session_state.mode == "RAG":
                    with chat_container:
                        with st.chat_message("assistant", avatar="🤖"):
                            stream_response = st.session_state.rag_info["query_engine"].query(user_input)
                            full_response = process_response_chunks(stream_response.response_gen)

                            sources = f"\n\n_Sources: {stream_response.source_nodes} documents_"
                            st.markdown(sources)
                            full_response += sources

                    st.session_state.chats[st.session_state.current_chat]["messages"].append({
                        "role": "assistant", 
                        "content": full_response
                    })
                else:
                    prepared_messages = prepare_context(
                        st.session_state.chats[st.session_state.current_chat]["messages"],
                        st.session_state.chats[st.session_state.current_chat]["system_messages"]
                    )
                    chat_history = [
                        ChatMessage(role=msg["role"], content=msg["content"])
                        for msg in prepared_messages
                    ]
                    with chat_container:
                        with st.chat_message("assistant", avatar="🤖"):
                            stream_response = st.session_state.llm.stream_chat(chat_history)
                            full_response = process_response_chunks(stream_response)
                    
                    st.session_state.chats[st.session_state.current_chat]["messages"].append({
                        "role": "assistant", 
                        "content": full_response
                    })
            st.session_state.processing = False
            save_chat_history()
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка обработки запроса: {str(e)}")
            st.session_state.processing = False
