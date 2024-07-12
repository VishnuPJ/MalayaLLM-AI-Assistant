import torch
import librosa
import numpy as np
import sounddevice as sd
from transformers import AutoModelForCTC, AutoProcessor
import tkinter as tk
from tkinter import ttk
import threading
import requests
import json
import queue
import sseclient


class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ഒരു നാടൻ ജാർവിസ്")
        self.root.geometry("500x700")
        self.root.configure(bg="#1b2b34")  # Dark bluish background
        self.audio_listen_time = 3
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.full_audio_data = []
        self.fs = 16000  # Sample rate

        # Load model and processor once
        self.model_id = "Bajiyo/w2v-bert-2.0-nonstudio_and_studioRecords_final"
        self.asr_processor = AutoProcessor.from_pretrained(self.model_id)
        self.asr_model = AutoModelForCTC.from_pretrained(self.model_id)

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Configure styles
        style.configure("TButton", padding=10, font=('Helvetica', 12), background="#343d46", foreground="#ffffff")
        style.map("TButton", background=[('active', '#ff9800')])
        style.configure("TLabel", background="#1b2b34", font=('Helvetica', 12), foreground="#ffffff")
        style.configure("TFrame", background="#22313f")
        style.configure("TLabelFrame", background="#22313f", font=('Helvetica', 12, 'bold'), foreground="#ffffff")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="20 20 20 20", style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="MalayaLLM(മലയാളം) AI Assistant ", font=('Helvetica', 18, 'bold'), background="#22313f", foreground="#ffffff")
        title_label.pack(pady=(0, 20))

        # Buttons frame
        button_frame = ttk.Frame(main_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Recording", command=self.start_recording, style="TButton")
        self.start_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED, style="TButton")
        self.stop_button.pack(side=tk.LEFT, expand=True, padx=5)

        self.submit_button = ttk.Button(button_frame, text="Submit", command=self.submit_audio, state=tk.DISABLED, style="TButton")
        self.submit_button.pack(side=tk.LEFT, expand=True, pady=5)

        # Status
        self.status_label = ttk.Label(main_frame, text="Status: Idle", foreground="#61dafb", background="#1b2b34")
        self.status_label.pack(pady=10)

        # Real-time Transcription
        transcription_frame = ttk.LabelFrame(main_frame, text="Malayalam Transcription : Bajiyo/w2v-bert-2.0-nonstudio_and_studioRecords_final", padding=2)
        transcription_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.transcription_text = tk.Text(transcription_frame, wrap=tk.WORD, height=8, font=('Helvetica', 11), background="#343d46", foreground="#ffffff")
        self.transcription_text.pack(fill=tk.BOTH, expand=True)

        # Response
        response_frame = ttk.LabelFrame(main_frame, text="Streaming Response : VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0_GGUF", padding=2)
        response_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.response_text = tk.Text(response_frame, wrap=tk.WORD, height=6, font=('Helvetica', 11), background="#343d46", foreground="#ffffff")
        self.response_text.pack(fill=tk.BOTH, expand=True)

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.full_audio_data = []
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.submit_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Recording...", foreground="#dc3545")
            self.transcription_text.delete(1.0, tk.END)
            self.response_text.delete(1.0, tk.END)
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            self.transcription_thread = threading.Thread(target=self.real_time_transcribe)
            self.transcription_thread.start()

    def record_audio(self):
        with sd.InputStream(samplerate=self.fs, channels=1, dtype='int16', callback=self.audio_callback):
            while self.is_recording:
                sd.sleep(100)

    def audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy())
        self.full_audio_data.append(indata.copy())

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.submit_button.config(state=tk.NORMAL)

            self.status_label.config(text="Status: Processing audio...", foreground="#61dafb")

            # Process the full audio
            full_audio = np.concatenate(self.full_audio_data)
            final_transcription = self.transcribe_audio(full_audio)

            # Update the transcription text with the final version
            self.transcription_text.delete(1.0, tk.END)
            self.transcription_text.insert(tk.END, final_transcription)

            self.status_label.config(text="Status: Stopped, ready to submit", foreground="#28a745")

    def real_time_transcribe(self):
        buffer = np.array([], dtype=np.int16)
        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=1)
                buffer = np.concatenate((buffer, data.flatten()))

                # Process every n seconds of accumulated audio
                while len(buffer) >= self.fs * self.audio_listen_time:
                    audio_segment = buffer[:self.fs * self.audio_listen_time]

                    # Process the accumulated segment
                    transcription = self.transcribe_audio(audio_segment)
                    self.update_transcription(transcription)

                    # Remove processed segment from buffer
                    buffer = buffer[self.fs * self.audio_listen_time:]

            except queue.Empty:
                continue

    def transcribe_audio(self, audio_data):
        # Convert audio to float32 and normalize
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        # Process the audio
        audio_array = librosa.resample(audio_data.flatten(), orig_sr=self.fs, target_sr=16000)

        inputs = self.asr_processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.asr_model(**inputs).logits

        predicted_ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.asr_processor.batch_decode(predicted_ids.unsqueeze(0))[0]
        return transcription

    def update_transcription(self, new_text):
        self.transcription_text.insert(tk.END, new_text + " ")
        self.transcription_text.see(tk.END)

    def submit_audio(self):
        final_transcription = self.transcription_text.get(1.0, tk.END).strip()
        if not final_transcription:
            self.status_label.config(text="Status: No transcription available!", foreground="#dc3545")
            return

        # Call the API and stream the response
        self.stream_api_response(final_transcription)

    def stream_api_response(self, transcription):
        url = "http://localhost:8080/completion"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        payload = {
            "prompt": f"ഒരു ചുമതല വിവരിക്കുന്ന ഒരു നിർദ്ദേശം ചുവടെയുണ്ട്. അഭ്യർത്ഥന ശരിയായി പൂർത്തിയാക്കുന്ന ഒരു പ്രതികരണം എഴുതുക. ### നിർദ്ദേശം:{transcription} ### പ്രതികരണം:",
            "stream": True
        }

        self.response_text.delete(1.0, tk.END)

        def stream_response():
            try:
                response = requests.post(url, headers=headers, json=payload, stream=True)
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.data:
                        try:
                            data = json.loads(event.data)
                            content = data.get('content', '')
                            if content:
                                self.root.after(0, self.update_response, content)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                self.root.after(0, self.update_response, f"Error: {str(e)}")

        threading.Thread(target=stream_response).start()

    def update_response(self, new_text):
        self.response_text.insert(tk.END, new_text)
        self.response_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()


