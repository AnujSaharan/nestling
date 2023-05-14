import os
import glob
import random
import re
import yt_dlp
import time
import whisper
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple
from pyannote.audio import Pipeline
from pyannote.core import Timeline
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.effects import compress_dynamic_range as compressor
from scipy.signal import butter, lfilter
from concurrent.futures import ThreadPoolExecutor


load_dotenv()  # Load environment variables from .env

def load_audio(audio_file_path: str) -> AudioSegment:
    return AudioSegment.from_wav(audio_file_path)

def apply_compression(audio: AudioSegment, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0) -> AudioSegment:
    return compressor(audio, threshold=threshold, ratio=ratio, attack=attack, release=release)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def apply_highpass_filter(audio: AudioSegment, cutoff=100) -> AudioSegment:
    audio_np = np.array(audio.get_array_of_samples()).astype(float)
    filtered_audio_np = highpass_filter(audio_np, cutoff, audio.frame_rate)

    # Cast the filtered NumPy array back to the original data type
    filtered_audio_np = filtered_audio_np.astype(audio.array_type)

    return AudioSegment(
        filtered_audio_np.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels,
    )

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
    model = whisper.load_model("medium")
    print("Transcribing audio file:", audio_file_path)
    start_time = time.time()
    
    result = model.transcribe(
        audio_file_path,
        word_timestamps=True,
        verbose=False,
        language="en",
        initial_prompt="NVIDIA Omniverse is a USD platform, a toolkit for building metaverse applications. We are building the Omniverse platform to enable 3D collaboration. Robotics is the wave of AI that's upcoming.",
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    
    print("Audio file transcribed in {:.2f} seconds".format(time.time() - start_time))

    # Write result to file
    audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    return result

def create_output_directories(audio_file_name: str) -> str:
    output_dir = os.path.join(audio_file_name + "_segments")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    return output_dir

def detect_silence(sound, position='leading', silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0
    assert chunk_size > 0
    audio_len = len(sound)
    
    if position == 'leading':
        while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < audio_len:
            trim_ms += chunk_size
    elif position == 'trailing':
        while sound[-trim_ms - chunk_size:-trim_ms].dBFS < silence_threshold and trim_ms < audio_len:
            trim_ms += chunk_size

    return min(trim_ms, audio_len)

def process_audio_segment(segment: AudioSegment) -> AudioSegment:
    segment = apply_compression(segment)
    segment = apply_highpass_filter(segment)
    return segment


def process_and_export_audio_segment(i, segment, output_dir):
    segment = process_audio_segment(segment)
    output_file_path = os.path.join(output_dir, "wavs", str(i + 1) + ".wav")
    segment.export(output_file_path, format="wav")


def split_audio_into_segments(audio: AudioSegment, sentences: List[Tuple[str, float, float]], output_dir: str, val_indices: List[int], create_val_set: bool = True, buffer_ms: int = 100) -> None:
    print("Splitting audio file into segments based on sentences")
    audio = audio.set_channels(1).set_frame_rate(
        22050)  # Set to mono and 22050Hz
    total_duration = 0
    segments = []
    for i, sentence in enumerate(sentences):
        sentence_text, start_time, end_time = sentence
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + buffer_ms  # Add buffer
        segment = audio[start_ms:end_ms]
        total_duration += len(segment)

        # Remove silence at the start of the segment
        leading_silence_end = detect_silence(segment, position='leading')
        trailing_silence_start = detect_silence(segment, position='trailing')
        segment = segment[leading_silence_end:len(
            segment) - trailing_silence_start]

        segments.append(segment)

    # Parallelize the processing and export of audio segments
    with ThreadPoolExecutor() as executor:
        executor.map(process_and_export_audio_segment, range(
            len(segments)), segments, [output_dir] * len(segments))

    return total_duration / 1000

def transcribe_audio(create_val_set: bool = True) -> None:
    start_time = time.time()

    for audio_file_path in glob.glob(os.path.join(".", "SPEAKER_*.wav")):
        original_duration = AudioSegment.from_wav(audio_file_path).duration_seconds
        print(f"Original duration of {os.path.basename(audio_file_path)}: {original_duration:.2f} seconds")

        result = transcribe_speaker_file(audio_file_path)

        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_dir = create_output_directories(audio_file_name)

        sentences = extract_sentences(result)
        val_indices = write_sentences_to_file(sentences, os.path.join(output_dir, "train.txt"), os.path.join(output_dir, "validation.txt"), create_val_set)

        audio = AudioSegment.from_file(audio_file_path)
        split_audio_into_segments(audio, sentences, output_dir, val_indices, create_val_set)

        # Calculate combined duration of the speaker segments
        segments_duration = 0
        for segment_path in glob.glob(os.path.join(output_dir, "wavs", "*.wav")):
            segment_audio = AudioSegment.from_wav(segment_path)
            segments_duration += segment_audio.duration_seconds
        
        print(f"Combined duration of {os.path.basename(audio_file_path)} segments: {segments_duration:.2f} seconds")
        print(f"Audio file split into segments in {time.time() - start_time:.2f} seconds")

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


def extract_audio_from_youtube(youtube_video_urls: List[str]) -> None:
    print("Extracting audio from YouTube videos")

    audio_download_options = {
        "format": "bestaudio/best",  # Choose the best audio quality available
        "outtmpl": "audio%(num)s.%(ext)s",  # Set the output file name format
        "postprocessors": [{
            "key": "FFmpegExtractAudio",  # Extract audio using FFmpeg
            "preferredcodec": "wav",  # Convert audio to WAV format
        }]
    }

    final_audio = None

    for index, youtube_video_url in enumerate(youtube_video_urls):
        audio_download_options["outtmpl"] = f"audio{index}.%(ext)s"

        try:
            with yt_dlp.YoutubeDL(audio_download_options) as youtube_downloader:
                youtube_downloader.download([youtube_video_url])

            current_audio = AudioSegment.from_wav(f"audio{index}.wav")
            normalized_audio = normalize(current_audio)  # Normalize the audio

            if final_audio is None:
                final_audio = normalized_audio
            else:
                final_audio = final_audio + normalized_audio

            os.remove(f"audio{index}.wav")

        except yt_dlp.utils.DownloadError as e:
            print(f"Error: {str(e)}")

    if final_audio is not None:
        final_audio.export("audio.wav", format="wav")
        print("All audio files concatenated and saved as audio.wav")

def main():
    youtube_urls_str = os.getenv('YOUTUBE_URLS')
    youtube_urls = youtube_urls_str.split(',')
    # extract_audio_from_youtube(youtube_urls)
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