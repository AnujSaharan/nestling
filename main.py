import os
import glob
import random
import re
import yt_dlp
import time
import whisper
from dotenv import load_dotenv
from typing import List, Tuple
from pyannote.audio import Pipeline
from pyannote.core import Timeline
from pydub import AudioSegment

load_dotenv()  # Load environment variables from .env

def load_audio(audio_file_path: str) -> AudioSegment:
    return AudioSegment.from_wav(audio_file_path)

def detect_overlapping_speech(audio_file_path: str) -> Timeline:
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

def remove_overlapping_bits(audio: AudioSegment, overlapping_segments: Timeline) -> AudioSegment:
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
    audio = audio.set_channels(1).set_frame_rate(22050)  # Set to mono and 22050Hz
    audio.export(audio_file_path, format="wav")

def diarize_audio(audio_file_path: str, mode: str = "concatenate") -> None:
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

        if duration_s >= 3.0:  # Ensure the segment is at least 3 seconds long
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
                segment_audio = segment_audio.set_channels(1).set_frame_rate(22050)  # Set to mono and 22050Hz
                segment_audio.export(os.path.join(speaker, f"{i}.wav"), format="wav")
    elif mode == "concatenate":
        for speaker, speaker_segments in segments.items():
            segment_audio = AudioSegment.empty()
            for i, segment in enumerate(speaker_segments):
                start_time_ms, end_time_ms = segment
                segment_audio = segment_audio + audio[start_time_ms:end_time_ms] + AudioSegment.silent(duration=2000)
            segment_audio = segment_audio.set_channels(1).set_frame_rate(22050)  # Set to mono and 22050Hz
            segment_audio.export(f"{speaker}.wav", format="wav")
    print(f"Total duration of audio: {len(audio) / 1000:.2f} seconds")

def extract_sentences(result: dict) -> List[Tuple[str, float, float]]:
    min_duration = 3.0
    max_duration = 15.0
    sentences = []

    def add_sentence(sentence, start_time, end_time):
        duration = end_time - start_time
        if duration < min_duration and sentences:
            prev_sentence, prev_start_time, prev_end_time = sentences[-1]
            combined_duration = end_time - prev_start_time

            if combined_duration <= max_duration:
                sentence = prev_sentence + ' ' + sentence
                start_time = prev_start_time
                sentences[-1] = (sentence.strip(), start_time, end_time)
            else:
                if duration >= min_duration:
                    sentences.append((sentence.strip(), start_time, end_time))
        elif min_duration <= duration <= max_duration:
            sentences.append((sentence.strip(), start_time, end_time))

    words = [(wd['word'].strip(), wd['start'], wd['end']) for seg in result['segments'] for wd in seg['words']]
    current_sentence = ''
    sentence_start_time = 0

    for i, (word, start_time, end_time) in enumerate(words):
        if i == 0:
            sentence_start_time = start_time

        current_sentence += word + ' '

        is_end_of_sentence = re.search(r'[.?!]\s*$', current_sentence)
        is_last_word = i == len(words) - 1

        if is_end_of_sentence or is_last_word:
            add_sentence(current_sentence, sentence_start_time, end_time)
            current_sentence = ''
            sentence_start_time = end_time

    return sentences

def transcribe_speaker_file(audio_file_path: str) -> dict:
    model = whisper.load_model("large")
    print("Transcribing audio file:", audio_file_path)
    start_time = time.time()
    result = model.transcribe(audio_file_path, word_timestamps=True, language="en", verbose=False)
    print("Audio file transcribed in {:.2f} seconds".format(time.time() - start_time))

    # Write result to file
    audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    return result

def create_output_directories(audio_file_name: str) -> str:
    output_dir = os.path.join(audio_file_name + "_segments")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    return output_dir

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0
    assert chunk_size > 0
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return min(trim_ms, len(sound))

def split_audio_into_segments(audio_file_path: str, sentences: List[Tuple[str, float, float]], output_dir: str, val_indices: List[int], create_val_set: bool = True, buffer_ms: int = 100) -> None:
    print("Splitting audio file into segments based on sentences")
    audio = AudioSegment.from_file(audio_file_path, format='wav')
    for i, sentence in enumerate(sentences):
        sentence_text, start_time, end_time = sentence
        output_file_path = os.path.join(output_dir, "wavs", str(i+1) + ".wav")
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + buffer_ms  # Add buffer
        segment = audio[start_ms:end_ms]
        segment = segment.set_channels(1).set_frame_rate(22050)  # Set to mono and 22050Hz

        # # Remove silence at the start of the segment
        # leading_silence_end = detect_leading_silence(segment)
        # segment = segment[leading_silence_end:]

        segment.export(output_file_path, format="wav")

def transcribe_audio(create_val_set: bool = True) -> None:
    start_time = time.time()

    for audio_file_path in glob.glob(os.path.join(".", "SPEAKER_*.wav")):
        result = transcribe_speaker_file(audio_file_path)

        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_dir = create_output_directories(audio_file_name)

        sentences = extract_sentences(result)
        val_indices = write_sentences_to_file(sentences, os.path.join(output_dir, "train.txt"), os.path.join(output_dir, "validation.txt"), create_val_set)

        split_audio_into_segments(audio_file_path, sentences, output_dir, val_indices, create_val_set)

        print("Audio file split into segments in {:.2f} seconds".format(time.time() - start_time))

    # Remove diarized long files after transcription is complete
    for audio_file_path in glob.glob(os.path.join(".", "SPEAKER_*.wav")):
        os.remove(audio_file_path)

def write_sentences_to_file(sentences: List[Tuple[str, float, float]], train_file_path: str, val_file_path: str, create_val_set: bool = True) -> List[int]:
    if create_val_set:
        random.shuffle(sentences)
        num_val = int(len(sentences) * 0.1)
        val_sentences = sentences[:num_val]
        train_sentences = sentences[num_val:]
    else:
        num_val = 0
        train_sentences = sentences

    val_indices = []
    with open(train_file_path, 'w') as f_train, open(val_file_path, 'w') as f_val:
        for i, sentence in enumerate(sentences):
            sentence_text, start_time, end_time = sentence
            output_file_name = str(i+1) + ".wav"
            sentence_text = sentence_text.replace('\n', ' ')
            line = f"wavs/{output_file_name}|{sentence_text}"
            
            if create_val_set and i < num_val:
                f_val.write(line + '\n')
                val_indices.append(i)
            else:
                if i == len(sentences) - 1:
                    f_train.write(line)
                else:
                    f_train.write(line + '\n')
    return val_indices

def extract_audio_from_youtube(youtube_video_url: str) -> None:
    print("Extracting audio from YouTube video")
    audio_download_options = {
    "format": "bestaudio/best", # Choose the best audio quality available
    "outtmpl": "audio.%(ext)s", # Set the output file name format
    "postprocessors": [{
    "key": "FFmpegExtractAudio", # Extract audio using FFmpeg
    "preferredcodec": "wav", # Convert audio to WAV format
    }]
    }
    try:
        with yt_dlp.YoutubeDL(audio_download_options) as youtube_downloader:
            youtube_downloader.download([youtube_video_url])
    except yt_dlp.utils.DownloadError as e:
        print(f"Error: {str(e)}")

def main():
    extract_audio_from_youtube(os.getenv('YOUTUBE_URL'))
    start_time = time.time()
    raw_audio_duration = len(load_audio(os.getenv('AUDIO_FILE_PATH'))) / 1000
    diarize_audio(os.getenv('AUDIO_FILE_PATH'))
    transcribe_audio()

    # Remove diarized long files after transcription is complete
    for audio_file_path in glob.glob(os.path.join(".", "*.wav")):
        os.remove(audio_file_path)

    end_time = time.time()
    real_time_factor = raw_audio_duration / (end_time - start_time)
    print(f"Real time factor: {real_time_factor:.2f}x")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()