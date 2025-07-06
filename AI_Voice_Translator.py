# Importamos las librerías necesarias
import gradio as gr                      # Para crear una interfaz web interactiva
import whisper                           # Para transcripción de audio a texto con IA
from translate import Translator         # Para traducir el texto a otros idiomas
from dotenv import dotenv_values         # Para cargar variables de entorno (API Keys)
from elevenlabs.client import ElevenLabs # Cliente para usar la API de ElevenLabs (TTS)
from elevenlabs import VoiceSettings     # Para configurar la voz generada por ElevenLabs

# Cargamos la API Key desde el archivo .env
config = dotenv_values('.env')
ELEVEN_LABS_API_KEY = config['ELEVEN_LABS_API_KEY']


# Función principal que se usará desde la interfaz

def translator(audio_file):

    # 1. TRANSCRIPCIÓN DEL AUDIO (español a texto usando Whisper)

    try:
        model = whisper.load_model('base')  # Cargamos el modelo base de Whisper
        result = model.transcribe(audio_file, language='Spanish', fp16=False)  # Transcribimos audio en español
        transcription = result['text']      # Extraemos el texto transcrito
    except Exception as e:
        raise gr.Error(f'Se ha producido un error transcribiendo el texto {str(e)}')
    
    print(f"Texto original: {transcription}")

    # 2. TRADUCCIÓN DEL TEXTO A VARIOS IDIOMAS

    try:
        en_transcription = Translator(from_lang='es', to_lang='en').translate(transcription)   # Inglés
        it_transcription = Translator(from_lang="es", to_lang="it").translate(transcription)   # Italiano
        ja_transcription = Translator(from_lang="es", to_lang="ja").translate(transcription)   # Japonés
        de_transcription = Translator(from_lang="es", to_lang="de").translate(transcription)   # Alemán

    except Exception as e:
        raise gr.Error(f'Se ha producido un error traduciendo el texto {str(e)}')
    
    # Mostramos los textos traducidos en consola

    print(f"Texto traducido a Inglés: {en_transcription}")
    print(f"Texto traducido a Italiano: {it_transcription}")
    print(f"Texto traducido a Japonés: {ja_transcription}")
    print(f"Texto traducido a Alemán: {de_transcription}")

    # 3. CONVERTIR TEXTO TRADUCIDO A AUDIO (con ElevenLabs)

    en_save_file_path = text_to_speach(en_transcription, "en")
    it_save_file_path = text_to_speach(it_transcription, "it")
    ja_save_file_path = text_to_speach(ja_transcription, "ja")
    de_save_file_path = text_to_speach(de_transcription, "de")

    # Devolvemos las rutas a los archivos de audio generados

    return en_save_file_path, it_save_file_path, ja_save_file_path, de_save_file_path


# Función que convierte texto a audio usando ElevenLabs

def text_to_speach(text: str, language: str) -> str:
    try:
        client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)  # Inicializamos el cliente con la API Key

        response = client.text_to_speech.convert(
            voice_id="bIHbv24MWmeRgasZH58o",              # ID de la voz a usar
            optimize_streaming_latency="0",               
            output_format="mp3_22050_32",                 
            text=text,                                    # Texto que se va a convertir a voz
            model_id="eleven_turbo_v2",                   
            voice_settings=VoiceSettings(                 
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        # Guardamos el audio generado por idioma en un archivo mp3

        save_file_path = f'{language}.mp3'  

        with open(save_file_path, 'wb') as f:
            for chuck in response:
                if chuck:
                    f.write(chuck)  # Escribimos el audio por partes

    except Exception as e:
        raise gr.Error(f'Se ha producido un error generando audio {str(e)}')

    return save_file_path  # Devolvemos la ruta del archivo generado


# INTERFAZ DE USUARIO (Gradio)

web = gr.Interface(
    fn=translator,                       # Función que se ejecuta
    inputs=gr.Audio(                     # Entrada de audio desde micrófono
        sources=['microphone'],
        type='filepath',
        label='Español'
    ),
    outputs=[                            # Salidas de audio traducido
        gr.Audio(label="Inglés"),
        gr.Audio(label="Italiano"),
        gr.Audio(label="Japonés"),
        gr.Audio(label="Alemán")
    ],
    title='Traductor de voz',           # Título de la interfaz
    description='Traductor de voz con IA a varios idiomas'  # Descripción visible en la app
)

# Lanzamos la app en el navegador
web.launch()
