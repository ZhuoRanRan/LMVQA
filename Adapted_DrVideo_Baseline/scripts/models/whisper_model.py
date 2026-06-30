# models/whisper_model.py — Whisper ASR wrapper (CPU by default)

import whisper

def has_intersection(t1, t2):
    return not (t1[1] < t2[0] or t2[1] < t1[0])

class AudioTranslator():
    def __init__(self, model='base', device='cpu'):
        # model: tiny/base/small/medium/large
        self.device = device
        self.model = whisper.load_model(model).to(device)

    def __call__(self, video_path):
        # Returns a list of segments, each with "start", "end", "text"
        segments = self.model.transcribe(video_path)["segments"]
        return segments
    
    def match(self, audio_results, start, end):
        transcript = ''
        for res in audio_results:
            if has_intersection((start, end), (res["start"], res["end"])):
                transcript += res['text'] + ' '
        return transcript
