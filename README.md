# AI Sales Assistant Chatbot
Please note that all the code provided in this document is pseudocode. I'm not able to run it because I am working on a Windows laptop, and the required dependencies for this project need a Linux/macOS environment with a working microphone, which I currently do not have access to.

Nevertheless, I have carefully reviewed the project and propose the following solution. You can find the corresponding pseudocode in `packages/worker.py` and `main.py`.

## How Latency is Introduced (originally)
The current workflow introduces latency in several stages, as follows:
1. The main thread starts processing a chunk of audio.
2. The main thread calls the STT model.
3. The main thread waits for the STT to complete (**Latency 1**).
4. The main thread calls the response generation model.
5. The main thread waits for the response generation to complete (**Latency 2**).
6. The main thread calls the TTS model.
7. The main thread waits for the TTS to complete (**Latency 3**).
8. The main thread starts processing the next audio chunk.

## How Latency is Reduced
To optimize performance, I introduced three threads: the main thread, a chatbot worker thread, and a speak worker thread. These threads run in parallel, and calls to response generation and TTS are now non-blocking. This means the main thread can continue processing STT tasks without waiting for response or TTS generation.

### Main Thread
The main thread handles the STT tasks.

1. It calls the STT model to get the transcription.
2. It sends the transcription to the chatbot worker.
3. The chatbot worker adds the transcription to its queue and returns immediately.
4. The main thread continues processing additional transcriptions without waiting for response generation or TTS to finish.

### Chatbot Worker
The chatbot worker is responsible for generating responses.

1. It constantly monitors the task queue.
2. When a new transcription arrives from the main thread, the chatbot worker calls the OpenAI API to generate a response.
3. Once a response is generated, it is passed to the speak worker.
4. The speak worker adds the response to its queue and returns immediately.
5. The chatbot worker can continue processing response generation tasks without waiting for TTS.

### Speak Worker
The speak worker handles the TTS tasks.

1. It constantly monitors the task queue.
2. When a new task arrives from the chatbot worker, the speak worker calls the TTS model to convert the response into speech.

## How to User Interruptions are Handled
If a user starts speaking while the chatbot is also speaking, the main thread will call the interrupt methods on both the speak worker and the chatbot worker. This clears their task queues, stopping the chatbot from speaking after it finishes the current sentence.

```python
if text != state["last_text"]:
    {...}

    # If the user has started speaking, interrupt the workers and clear their task queues
    speak_worker.interrupt()
    chatbot_worker.interrupt()
```

## (Potentially) Improving STT Inference Speed with FP16
To further reduce latency in the STT inference process, we can use automatic mixed precision using PyTorch’s autocast feature. By enabling autocast, we can perform STT model inference faster, especially when working with large neural networks, by using lower precision (FP16) without sacrificing much accuracy.

```python
class StreamingTranscription:
    {...}
    def _load_model(self, model_name, lookahead_size, decoder_type):
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name).to("cuda")
        {...}
    
    def transcribe_chunk(self, new_chunk):
        {...}

        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                (
                    self.pred_out_stream,
                    transcribed_texts,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_len,
                    self.previous_hypotheses,
                ) = self.asr_model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.pred_out_stream,
                    drop_extra_pre_encoded=None,
                    return_transcription=True,
                )
        {...}