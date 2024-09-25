import threading
import time

from elevenlabs_tts import speak
from packages.sales_chatbot import SalesChatbot

class SpeakWorker(threading.Thread):
    def __init__(self, task_queue):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.is_speaking = False

    def run(self):
        """
        Continuously checking the task queue for new AI response.
        When an response is received, use TTS to speak it.
        """
        print("Speak worker started")
        while True:
            text = self.task_queue.get()
            start_time = time.time()

            print(f"SPEAK: {text}")
            self.speak(text)
            print(f"TTS latency: {time.time() - start_time}s")
            
            self.task_queue.task_done()

    def speak(self, text):
        self.is_speaking = True
        speak(text)
        self.is_speaking = False

    def add_speak_task(self, text):
        self.task_queue.put(text)
    
    def interupt(self):
        """
        Clears the task queue, stopping any ongoing tasks.
        """
        with self.task_queue.mutex:
            self.task_queue.queue.clear()

        self.is_speaking = False

class ChatbotWorker(threading.Thread):
    def __init__(self, task_queue, speak_worker):
        super().__init__(daemon=True)
        self.chatbot = SalesChatbot()
        self.task_queue = task_queue
        self.speak_worker = speak_worker
        

    def run(self):
        """
        Continuously checking the task queue for new text input.
        When text is received, generates a response and adds it to the speak worker's task queue.
        """
        print("Chatbot worker started")
        while True:
            text = self.task_queue.get()

            start_time = time.time()
            ai_response = self.generate_response(text)
            print(f"Response generation latency: {time.time() - start_time}s")

            self.speak_worker.add_speak_task(ai_response)
            
            self.task_queue.task_done()

    def generate_response(self, text):
        ai_response = self.chatbot.generate_response(text)
        print(f"AI: {ai_response}")
        return ai_response

    def add_chatbot_task(self, text):
        self.task_queue.put(text)

    def interupt(self):
        """
        Clears the task queue, stopping any ongoing tasks.
        """
        with self.task_queue.mutex:
            self.task_queue.queue.clear()