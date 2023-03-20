import os
import glob
import yt_dlp
import time
import whisper
from dotenv import load_dotenv
from typing import List, Tuple
from pyannote.audio import Pipeline
from pyannote.core import Timeline
from pydub import AudioSegment

load_dotenv() # Load environment variables from .env

def load_audio(audio_file_path: str) -> AudioSegment:
    return AudioSegment.from_wav(audio_file_path)

def detect_overlapping_speech(audio_file_path: str) -> List:
    print("Detecting overlapping speech")
    start_time = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection", use_auth_token=os.getenv('HUGGINGFACE_API_KEY'))
    output = pipeline(audio_file_path)
    overlapping_segments = output.get_timeline()
    overlapping_duration = overlapping_segments.duration()
    end_time = time.time()
    print(f"Overlapping speech detection took {end_time - start_time:.2f} seconds")
    print(f"Total overlapping duration: {overlapping_duration:.2f} seconds")
    return overlapping_segments

def remove_overlapping_bits(audio: AudioSegment, overlapping_segments: List) -> AudioSegment:
    print("Removing overlapping speech")
    start_time = time.time()
    timeline = Timeline(overlapping_segments)
    non_overlapping_audio = AudioSegment.empty()
    buffer = []
    prev_end_time_ms = 0
    for _, speech in enumerate(timeline.support()):
        if speech.duration > 0:
            start_time_ms = int(speech.start * 1000)
            end_time_ms = int(speech.end * 1000)
            buffer.append(audio[prev_end_time_ms:start_time_ms])
            prev_end_time_ms = end_time_ms
        if len(buffer) >= 10:
            non_overlapping_audio += AudioSegment.silent(duration=1)
            non_overlapping_audio = non_overlapping_audio[:-1] + sum(buffer)
            buffer = []
    # Append any remaining audio after the last overlapping segment
    buffer.append(audio[prev_end_time_ms:])
    if buffer:
        non_overlapping_audio += AudioSegment.silent(duration=1)
        non_overlapping_audio = non_overlapping_audio[:-1] + sum(buffer)
    end_time = time.time()
    print(f"Overlapping speech removal took {end_time - start_time:.2f} seconds")
    print(f"Total duration of non-overlapping audio: {len(non_overlapping_audio) / 1000:.2f} seconds")
    return non_overlapping_audio

def save_cleaned_audio(audio: AudioSegment, audio_file_path: str) -> None:
    audio.export(audio_file_path, format="wav")

def diarizeAudio(audio_file_path: str, mode: str = "concatenate") -> None:
    audio = load_audio(audio_file_path)
    overlapping_segments = detect_overlapping_speech(audio_file_path)
    audio_without_overlap = remove_overlapping_bits(audio, overlapping_segments)
    save_cleaned_audio(audio_without_overlap, audio_file_path)
    print("Starting speaker diarization")
    start = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv('HUGGINGFACE_API_KEY'))
    diarization = pipeline(audio_file_path)

    segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time_ms = int(turn.start * 1000)
        end_time_ms = int(turn.end * 1000)
        duration_s = (end_time_ms - start_time_ms) / 1000

        if speaker not in segments:
            segments[speaker] = []
        if duration_s >= 4.0:
            segments[speaker].append((start_time_ms, end_time_ms))

    end = time.time()
    print(f"Speaker diarization took {end - start:.2f} seconds")

    audio = load_audio(audio_file_path)
    if mode == "segments":
        for speaker, speaker_segments in segments.items():
            if not os.path.exists(speaker):
                os.makedirs(speaker)
            for i, segment in enumerate(speaker_segments):
                start_time_ms, end_time_ms = segment
                segment_audio = audio[start_time_ms:end_time_ms]
                segment_audio.export(os.path.join(speaker, f"{i}.wav"), format="wav")
    elif mode == "concatenate":
        for speaker, speaker_segments in segments.items():
            segment_audio = AudioSegment.empty()
            for i, segment in enumerate(speaker_segments):
                start_time_ms, end_time_ms = segment
                segment_audio = segment_audio + audio[start_time_ms:end_time_ms] + AudioSegment.silent(duration=2000)
            segment_audio.export(f"{speaker}.wav", format="wav")
    print(f"Total duration of audio: {len(audio) / 1000:.2f} seconds")

def extract_sentences(result: dict) -> List[Tuple[str, float, float]]:
    source_text = result['text']
    source_words = source_text.split()
    current_sentence = ''
    sentence_start_time = 0
    sentence_end_time = 0
    current_word_index = 0
    sentences = []
    min_duration = 3.0

    for seg in result['segments']:
        seg_text = seg['text'].strip()
        seg_words = seg_text.split()

        for seg_word in seg_words:
            if seg_word == source_words[current_word_index]:
                if current_word_index == 0:
                    sentence_start_time = seg['start']
                current_word_index += 1
                current_sentence += seg_word + ' '

                if seg_word[-1] in ['.', '?', '!'] or current_word_index == len(source_words):
                    sentence_end_time = seg['end']
                    duration = sentence_end_time - sentence_start_time

                    if duration >= min_duration or len(sentences) == 0:
                        sentences.append((current_sentence.strip(), sentence_start_time, sentence_end_time))
                    else:
                        merged_sentence, prev_start_time, prev_end_time = sentences[-1]
                        merged_sentence = f"{merged_sentence} {current_sentence.strip()}"
                        sentences[-1] = (merged_sentence, prev_start_time, sentence_end_time)

                    current_sentence = ''
                    if current_word_index < len(source_words):
                        sentence_start_time = seg['end']
            else:
                current_sentence += seg_word + ' '
    return sentences

def transcribeAudio() -> None:
    start_time = time.time()
    model = whisper.load_model("large")

    for audio_file_path in glob.glob(os.path.join(".", "*.wav")):
        print("Transcribing audio file:", audio_file_path)
        result = model.transcribe(audio_file_path, word_timestamps=True, language="en", verbose=True)
        print("Audio file transcribed in {:.2f} seconds".format(time.time() - start_time))

        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_dir = audio_file_name + "_segments"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
        write_sentences_to_file(extract_sentences(result), os.path.join(output_dir, "train.txt"))

        print("Splitting audio file into segments based on sentences")
        sentences = extract_sentences(result)
        for i, sentence in enumerate(sentences):
            sentence_text, start_time, end_time = sentence
            output_file_path = os.path.join(output_dir, "wavs", str(i+1) + ".wav")
            split_audio_file(audio_file_path, output_file_path, start_time, end_time)
        print("Audio file split into segments in {:.2f} seconds".format(time.time() - start_time))

        # Remove the original audio file
        os.remove(audio_file_path)

def write_sentences_to_file(sentences: List[Tuple[str, float, float]], output_file_path: str) -> None:
    with open(output_file_path, 'w') as f:
        for i, sentence in enumerate(sentences):
            sentence_text, start_time, end_time = sentence
            output_file_name = f"{i+1}.wav"
            f.write(f"wavs/{output_file_name}|{sentence_text}\n")

def split_audio_file(input_file_path: str, output_file_path: str, start_time: float, end_time: float) -> None:
    buffer_duration_ms = 150
    audio = AudioSegment.from_file(input_file_path, format='wav')
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000) + buffer_duration_ms
    segment = audio[start_ms:end_ms]
    segment.export(output_file_path, format="wav")
  
def extractAudioFromYouTube(youtubeVideoURL: str) -> None:
    print("Extracting audio from YouTube video")
    audioDownloadOptions = {
        "format": "bestaudio/best",  # Choose the best audio quality available
        "outtmpl": "audio.%(ext)s",  # Set the output file name format
        "postprocessors": [{
            "key": "FFmpegExtractAudio",  # Extract audio using FFmpeg
            "preferredcodec": "wav",  # Convert audio to WAV format
        }]
    }
    try:
        with yt_dlp.YoutubeDL(audioDownloadOptions) as youtubeDownloader:
            youtubeDownloader.download([youtubeVideoURL])
    except yt_dlp.utils.DownloadError as e:
        print(f"Error: {str(e)}")

def main():
    extractAudioFromYouTube(os.getenv('YOUTUBE_URL'))
    start_time = time.time()
    raw_audio_duration = len(load_audio(os.getenv('AUDIO_FILE_PATH'))) / 1000
    diarizeAudio(os.getenv('AUDIO_FILE_PATH'))
    os.remove(os.getenv('AUDIO_FILE_PATH'))
    transcribeAudio()
    end_time = time.time()
    real_time_factor = raw_audio_duration / (end_time - start_time)
    print(f"Real time factor: {real_time_factor:.2f}x")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()