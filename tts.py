#kerak bo`lgan liblar:
# pip install silero
# pip install sounddevice
# pip install numpy
import flet as ft
import torch
import sounddevice as sd
import time

def main(page: ft.Page):
    language = "uz"
    model_id = 'v3_uz'
    sample_rate = 48000
    speaker = 'dilnavoz'
    put_accent = True
    put_yoo = True
    device = torch.device('cpu') ###### Agar videokarta kuchli bo'sa shuni 'gpu' ga almashtirsa tezro sintez bo'ladi
    #text = "Assalomu aleykum xo'jayin! Amringizga muntazirman!" ####### Mana shu yerga kiritilgan text o`qiladi
    field = ft.TextField(width=500)

    def sintez(e):
        text = field.value
        print(len(text))
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
        model.to(device)

        audio = model.apply_tts(text=text,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent)

        print(text)
        sd.play(audio, sample_rate)
        time.sleep(len(audio) / sample_rate)
        sd.stop()
        field.value = ""
        page.update()
    btn = ft.ElevatedButton(text="Aytish", on_click=sintez)

    page.add(ft.Text("Matin"), field, btn)
ft.app(target=main)