import openai
import gradio as gr
import os
import psycopg2
from urllib.parse import urlparse

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database URL
DATABASE_URL = os.getenv('DATABASE_URL')

# Parse the database URL
result = urlparse(DATABASE_URL)
db_username = result.username
db_password = result.password
database = result.path[1:]
hostname = result.hostname
port = result.port

# Connect to the database
def connect_db():
    try:
        conn = psycopg2.connect(
            dbname=database,
            user=db_username,
            password=db_password,
            host=hostname,
            port=port
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Create table if not exists
def create_table():
    conn = connect_db()
    if conn:
        cur = conn.cursor()
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_transcripts (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            transcript TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        '''
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        conn.close()

# Store chat transcript
def store_transcript(user_id, transcript):
    conn = connect_db()
    if conn:
        cur = conn.cursor()
        insert_query = '''
        INSERT INTO chat_transcripts (user_id, transcript)
        VALUES (%s, %s);
        '''
        cur.execute(insert_query, (user_id, transcript))
        conn.commit()
        cur.close()
        conn.close()

# Register new user
def register_user(username, password):
    conn = connect_db()
    if conn:
        cur = conn.cursor()
        insert_query = '''
        INSERT INTO users (username, password)
        VALUES (%s, %s) RETURNING id;
        '''
        cur.execute(insert_query, (username, password))
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return user_id
    return None

# Authenticate user
def authenticate(username, password):
    conn = connect_db()
    if conn:
        cur = conn.cursor()
        select_query = '''
        SELECT id FROM users WHERE username = %s AND password = %s;
        '''
        cur.execute(select_query, (username, password))
        user_id = cur.fetchone()
        cur.close()
        conn.close()
        if user_id:
            return user_id[0]
    return None

# Initialize the table
create_table()

# Initial system messages for the chatbot
# NOTE: Change this to modify the chatbot behavior to fit your research needs
initial_messages = [
    {"role": "system", "content": "You are an attachment and close relationship research surveyor"},
    {"role": "user", "content": """ask me each question from this questionnaire and rewrite it as an open ended question and wait for each response. Empathize with me and regularly ask for clarification why I answered with a certain response. Here is the questionnaire:  
Can you describe your relationship with your mother or a mother-like figure in your life?
Do you usually discuss your problems and concerns with your mother or a mother-like figure?
"""},
]

def chatbot(input, state):
    user_id = state[0]
    messages = state[1]
    
    if input:
        messages.append({"role": "user", "content": input})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        
        conversation = ""
        for message in messages[2:]:
            role = "You" if message["role"] == "user" else "AttachmentBot"
            conversation += f"{role}: {message['content']}\n"

        store_transcript(user_id, conversation)
        return conversation, [user_id, messages]

with gr.Blocks() as demo:
    username = gr.Textbox(label="Username")
    password = gr.Textbox(label="Password", type="password")
    login_button = gr.Button("Login")
    register_button = gr.Button("Register")
    auth_message = gr.Textbox(visible=False)

    chat_input = gr.Textbox(lines=7, label="Chat with AttachmentBot", visible=False)
    chat_output = gr.Textbox(label="Conversation", visible=False)
    state = gr.State([None, initial_messages.copy()])

    def login(username, password):
        user_id = authenticate(username, password)
        if user_id:
            return gr.update(visible=True), gr.update(visible=True), [user_id, initial_messages.copy()], ""
        else:
            return gr.update(visible=False), gr.update(visible=False), [None, initial_messages.copy()], "Invalid credentials"

    def register(username, password):
        user_id = register_user(username, password)
        if user_id:
            return gr.update(visible=True), gr.update(visible=True), [user_id, initial_messages.copy()], "Registration successful, you can now login."
        else:
            return gr.update(visible=False), gr.update(visible=False), [None, initial_messages.copy()], "Registration failed, try a different username."

    login_button.click(login, inputs=[username, password], outputs=[chat_input, chat_output, state, auth_message])
    register_button.click(register, inputs=[username, password], outputs=[chat_input, chat_output, state, auth_message])
    # NOTE: Change this to change the user interface to fit your need
    chat_interface = gr.Interface(
        fn=chatbot,
        inputs=[chat_input, state],
        outputs=[chat_output, state],
        title="AttachmentBot",
        description="Let me survey you about your attachment with certain people in your life. To begin, enter 'start'."
    )

    demo.launch()
