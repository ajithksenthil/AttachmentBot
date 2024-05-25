import openai
import gradio as gr
import os

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initial system messages for the chatbot
initial_messages = [
    {"role": "system", "content": "You are an attachment and close relationship research surveyor"},
    {"role": "user", "content": """ask me each question from this questionnaire and rewrite it as an open ended question and wait for each response. Empathize with me and regularly ask for clarification why I answered with a certain response. Here is the questionnaire:  
Can you describe your relationship with your mother or a mother-like figure in your life?
Do you usually discuss your problems and concerns with your mother or a mother-like figure?
"""},
]

def chatbot(input):
    if not hasattr(chatbot, "messages"):
        chatbot.messages = initial_messages.copy()
    
    if input:
        chatbot.messages.append({"role": "user", "content": input})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chatbot.messages
        )
        reply = response.choices[0].message.content
        chatbot.messages.append({"role": "assistant", "content": reply})
        
        conversation = ""
        for message in chatbot.messages[2:]:
            role = "You" if message["role"] == "user" else "AttachmentBot"
            conversation += f"{role}: {message['content']}\n"

        return conversation

inputs = gr.Textbox(lines=7, label="Chat with AttachmentBot")
outputs = gr.Textbox(label="Conversation")

gr.Interface(
    fn=chatbot, 
    inputs=inputs, 
    outputs=outputs, 
    title="AttachmentBot",
    description="Let me survey you about your attachment with certain people in your life. To begin, enter 'start'."
).launch()