import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

X = torch.arange(-5,5,0.1)
w = torch.tensor(-12.0, requires_grad=True)

y = 3 * X

Y = y + 0.5 * torch.randn(X.size())


def forward(x):
    return w * x

def criterion(yhat,y):
    return torch.mean((yhat - y)**2)


def train_model(iteration_counter, lr):
    w_history = []
    loss_history = []

    for epoch in range(iteration_counter):
        Yhat = forward(X)
        loss = criterion(Yhat, y)  # Вычисляем ошибку

        loss.backward()

        w.data = w.data - lr * w.grad  # Обновляем параметр w
        w.grad.zero_()  # Обнуляем градиент перед следующей итерацией

        w_history.append(w.item())
        loss_history.append(loss.item())

        print(f"Иттерация {epoch+1}: w = {w.item()}, Ошибка = {loss.item()}")

        if loss.item() < 1e-60:
            break
    return w_history, loss_history


w_history, loss_history = train_model(50, lr=0.05)

plt.figure(figsize=(12, 5))

# График изменения w
plt.subplot(1, 2, 1)
plt.plot(w_history, label="w в процессе обучения")
plt.axhline(y=-980, color='r', linestyle='--', label="Истинное w = 3")
plt.xlabel("Итерации")
plt.ylabel("Значение w")
plt.title("Сходимость w")
plt.legend()

# График уменьшения ошибки
plt.subplot(1, 2, 2)
plt.plot(loss_history, label="Ошибка MSE")
plt.xlabel("Итерации")
plt.ylabel("Ошибка")
plt.title("Уменьшение ошибки")
plt.legend()

plt.show()


from openai import OpenAI
from flask import Flask, request
from langdetect import detect
import requests
import json
import os

client = OpenAI()

ASSISTANT_ID = os.getenv('ASSISTANT_ID')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Loading current users from file
with open("data.json", "r") as file:
    user_ids = json.load(file)  
    # Loading the file contents into the user_ids variable


def thread_setup(userID):
    """ The function returns the user's thread ID """

    if userID not in user_ids:
        #  Creating a new thread if user is not in the list
        empty_thread = client.beta.threads.create()
        user_ids[userID] = empty_thread.id
        with open(
            "data.json", "w"
        ) as file:  # Writing the thread ID to the file if the user was not there
            json.dump(user_ids, file)
        return empty_thread.id
    else:
        return user_ids[userID]

def add_a_message_to_the_Thread(user_message, thread_id):
    """" Adding the user's message to the thread """

    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_message
    )

def generate_response(thread_id, language="ru"):
    """ Generating the model's response """

    if language == "de":
        global ASSISTANT_ID
        ASSISTANT_ID = os.getenv('ASSISTANT_ID_DE')
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        )
    
    if run.status == "completed":
        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc"
            )
        # Sorting the thread so that the last message is from the bot
        return (
            messages.data[0].content[0].text.value
        )  # Returns the last message from the model
    else:
        print("Run status: ", run.status)

def send_message(chat_id, text, thought=False):
    """ Function for sending messages via Telegram API """

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    message = {"chat_id": chat_id, "text": text}
    headers = {"Content-Type": "application/json"}
    data = requests.post(url, json=message, headers=headers)
    data = data.json()
    if thought:
        return data["result"][
            "message_id"
        ]  # Sends message_id for interaction with the edit_message() function

def edit_message(chat_id, message_id, new_text):
    """ Edits the text of the message sent by the bot. """

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
    new_message = {"chat_id": chat_id, "message_id": message_id, "text": new_text}
    requests.post(url, json=new_message)

def detect_language(text):
    """ Determines the language of the given text """

    try:
        language = detect(text)
        return language
    except Exception as e:
        return f"Error while determining the language: {e}"
    

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # Retrieving JSON data from Telegram
        data = request.json
        chat_id = data["message"]["chat"]["id"]

        if "message" not in data or "text" not in data["message"]:
            send_message(chat_id, "Я принимаю только текст")
            return {"status": "ok"}, 200

        text = str(data["message"]["text"])
        userID = str(data["message"]["from"]["id"])
        language = detect_language(text)

        # Checking if a thread already exists for the user
        user_thread_id = thread_setup(userID)

        if language == "de":
            bot_message_id = send_message(chat_id, "Antwort wird generiert...", True)
        else:
            bot_message_id = send_message(chat_id, "Генерирую ответ...", True)

        # If it is user's first contact
        if text == "/start":
            send_message(
                chat_id,
                "Напиши, что ты хочешь сказать своему собеседнику и я дам рекомендации, "
                "как сказать это так чтобы он не обиделся.\n\n"
                "Schreiben Sie, was Sie Ihrem Gesprächspartner mitteilen möchten, und "
                "ich gebe Ihnen Empfehlungen, wie Sie es höflich ausdrücken können."
                )
        else:
            add_a_message_to_the_Thread(text, user_thread_id)
            answer = generate_response(user_thread_id, language)

            # Sending the generated response to the user
            if bot_message_id:
                edit_message(chat_id, bot_message_id, answer)
                bot_message_id = None
            else:
                send_message(chat_id, answer)
        return {"status": "ok"}, 200
    except Exception as e:
        print("Processing error:", e)
        return {"status": "error"}, 500

# Starts the Flask server on port 8443
app.run(port=8443)
