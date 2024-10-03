import os

import time
import queue
import pyaudio as pa
import numpy as np
from packages.nemo_stt import StreamingTranscription  # Update the class name if different # Update the class name if different
from packages.workers import SpeakWorker, ChatbotWorker


SAMPLE_RATE = 16000
CHUNK_SIZE = 160  # ms
WAIT_TIME = 500  # ms

transcriber = StreamingTranscription()
speak_worker = SpeakWorker(queue.Queue())
chatbot_worker = ChatbotWorker(queue.Queue(), speak_worker)

state = {
    "last_text": "",
    "silence_duration": 0,
}

def get_filter_response(text):
    filter_responses = ["Ummm.", "OK.", "Thanks for the info.", "Got it.", "I see.", "Right."]

    #in the future, the choice of filter responses can be based on the user's input. 
    #This can be done by a small neural network running on the local machine.
    return filter_responses[np.random.randint(0, len(filter_responses) - 1)]

def callback(in_data, *_):
    signal = np.frombuffer(in_data, dtype=np.int16)

    start_time = time.time()
    text = transcriber.transcribe_chunk(signal)
    print(f"STT latency: {time.time() - start_time}s")
    
    if text != state["last_text"]:
        state["last_text"] = text
        state["silence_duration"] = 0

        speak_worker.interupt()
        chatbot_worker.interupt()
    else:
        state["silence_duration"] += CHUNK_SIZE / 2
        
        # Check if the user said anything and at least WAIT_TIME has since passed
        if state["silence_duration"] >= WAIT_TIME and len(state["last_text"]) > 0:
            print(f"USER: {state['last_text']}")      

            speak_worker.add_speak_task(get_filter_response(state["last_text"]))

            # Add the user's last text to the chatbot task queue
            chatbot_worker.add_chatbot_task(state["last_text"])
            
            # Reset transcription cache and keep transcribing
            transcriber.reset_transcription_cache() 
            state["last_text"] = ""

    # Transcription processing is paused while the AI response is being generated, continue now
    return (in_data, pa.paContinue)

p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        print('Please type input device ID:')
        dev_idx = int(input())

    stream = p.open(format=pa.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=dev_idx,
        stream_callback=callback,
        frames_per_buffer=int(SAMPLE_RATE * CHUNK_SIZE / 1000) - 1
    )

    speak_worker.start()
    chatbot_worker.start()

    print('Listening...')

    stream.start_stream()
    
    try:
        while stream.is_active():
            time.sleep(0.1)
    finally:        
        stream.stop_stream()
        stream.close()
        p.terminate()

        print()
        print("PyAudio stopped")
else:
    print('ERROR: No audio input device found.')
