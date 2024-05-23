import nltk
from nltk.chat.util import Chat, reflections

# Download necessary NLTK data
nltk.download('punkt')

pairs = [
    [
        r"my name is (.*)",
        ["Hello %1! How can I assist you today?",]
    ],
    [
        r"what is your name?",
        ["You can call me ChatBot. How can I help you?",]
    ],
    [
        r"how are you?",
        ["I'm just a computer program, but I'm here to help you!",]
    ],
    [
        r"(.*) (hungry|tired|sleepy)",
        ["I'm just a machine, so I don't experience those feelings.",]
    ],
    [
        r"(.*)",
        ["I'm sorry, I don't understand. Can you please rephrase?",]
    ]
]

def main():
    print("Hi! I'm a simple AI chatbot. You can start a conversation with me. Type 'exit' to end the conversation.")
    chatbot = Chat(pairs, reflections)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break
        response = chatbot.respond(user_input)
        print("ChatBot:", response)

if __name__ == "__main__":
    main()
