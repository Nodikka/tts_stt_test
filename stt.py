import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import sys
import json
import torch
import time

'''This script processes audio input from the microphone and displays the transcribed text.'''

language = 'uz'
model_id = 'v3_uz'
sample_rate = 48000
speaker = 'dilnavoz'
put_accent = True
device = torch.device('cpu')

# list all audio devices known to your system
print("Display input/output devices")
print(sd.query_devices())


# get the samplerate - this is needed by the Kaldi recognizer
device_info = sd.query_devices(sd.default.device[0], 'input')
samplerate = int(device_info['default_samplerate'])

# display the default input device
print("===> Initial Default Device Number:{} Description: {}".format(sd.default.device[0], device_info))

# setup queue and callback function
q = queue.Queue()

def recordCallback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))
    
# build the model and recognizer objects.
print("===> Build the model and recognizer objects.  This will take a few minutes.")
model = Model("model/uzbek-tdnn")
recognizer = KaldiRecognizer(model, samplerate)
recognizer.SetWords(False)

print("===> Begin recording. Press Ctrl+C to stop the recording ")
try:
    with sd.RawInputStream(dtype='int16',
                           channels=1,

                           callback=recordCallback):
        while True:
            data = q.get()        
            if recognizer.AcceptWaveform(data):
                recognizerResult = recognizer.FinalResult()
                text = recognizer.Result()
                # convert the recognizerResult string into a dictionary  
                resultDict = json.loads(recognizerResult)
                if not resultDict.get("text", "") == "":
                    model1, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                              model='silero_tts',
                                              language=language,
                                              speaker=model_id)
                    model1.to(device)

                    audio = model1.apply_tts(text=resultDict["text"],
                                            speaker=speaker,
                                            sample_rate=sample_rate,
                                            put_accent=put_accent)
                    print(resultDict["text"])
                    sd.play(audio, sample_rate)
                    time.sleep(len(audio) / sample_rate)
                    sd.stop()
                else:
                    print("no input sound")

except KeyboardInterrupt:
    print('===> Finished Recording')
except Exception as e:
    print(str(e))