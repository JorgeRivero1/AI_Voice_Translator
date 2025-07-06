import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# Contraseña API
config = dotenv_values('.env')
ELEVEN_LABS_API_KEY = config['ELEVEN_LABS_API_KEY']


def translator(audio_file):

    # 1. Transcribimos el texto (Whisper)

    try:
        model = whisper.load_model('base')
        result = model.transcribe(audio_file, language='Spanish', fp16 = False)
        transcription = result['text']

    except Exception as e:
        raise gr.Error(
            f'Se ha producido un error transcribiendo el texto {str(e)}')
    
    print(f"Texto original: {transcription}")

    # 2. Traducimos texto (Translator)
    try:
        en_transcription = Translator(
            from_lang='es', to_lang='en').translate(transcription)
        it_transcription = Translator(
            from_lang="es", to_lang="it").translate(transcription)
        ja_transcription = Translator(
            from_lang="es", to_lang="ja").translate(transcription)
        de_transcription = Translator(
            from_lang="es", to_lang="de").translate(transcription)
        
    except Exception as e:
        raise gr.Error(
            f'Se ha producido un error traduciendo el texto {str(e)}')
    
    print(f"Texto traducido a Inglés: {en_transcription}")
    print(f"Texto traducido a Italiano: {it_transcription}")
    print(f"Texto traducido a Japonés: {ja_transcription}")
    print(f"Texto traducido a Alemán: {de_transcription}")


    # 3. Generar audio traducido

    en_save_file_path = text_to_speach(en_transcription, "en")
    it_save_file_path = text_to_speach(it_transcription, "it")
    ja_save_file_path = text_to_speach(ja_transcription, "ja")
    de_save_file_path = text_to_speach(de_transcription, "de")

    return en_save_file_path, it_save_file_path, ja_save_file_path, de_save_file_path

def text_to_speach(text: str, language: str) -> str:

    try:
        client = ElevenLabs(
            api_key=ELEVEN_LABS_API_KEY
        )

        response = client.text_to_speech.convert(
            voice_id="bIHbv24MWmeRgasZH58o",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = f'{language}.mp3'  

        with open(save_file_path, 'wb') as f:
            for chuck in response:
                if chuck:
                    f.write(chuck)

    except Exception as e:
        raise gr.Error(
            f'Se ha producido un error generando audio {str(e)}')
    return save_file_path
    

# Cramos la interfaz (Gradio)

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=['microphone'],
        type='filepath',
        label='Español'
    ),
    outputs=[
        gr.Audio(label="Inglés"),
        gr.Audio(label="Italiano"),
        gr.Audio(label="Japonés"),
        gr.Audio(label="Alemán")
    ],
    title='Traductor de voz',
    description='Traductor de voz con IA a varios idiomas'
)

web.launch()
