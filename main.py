import os
import glob
import youtube_dl
import time
import whisper
from dotenv import load_dotenv
from typing import List
from pyannote.audio import Pipeline
from pyannote.core import Timeline
from pydub import AudioSegment

load_dotenv() # Load environment variables from .env

def load_audio(audio_file_path: str) -> AudioSegment:
    return AudioSegment.from_wav(audio_file_path)

def detect_overlapping_speech(audio: AudioSegment, audio_file_path: str) -> List:
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
    overlapping_segments = detect_overlapping_speech(audio, audio_file_path)
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
        with youtube_dl.YoutubeDL(audioDownloadOptions) as youtubeDownloader:
            youtubeDownloader.download([youtubeVideoURL])
    except youtube_dl.utils.DownloadError as e:
        print(f"Error: {str(e)}")


def transcribeAudio() -> None:
    start_time = time.time()
    model = whisper.load_model("large")

    for audio_file_path in glob.glob(os.path.join(".", "*.wav")):
        print("Transcribing audio file:", audio_file_path)
        start_time = time.time()
        result = model.transcribe(audio_file_path, word_timestamps=True, language="en")
        print("Keys in result:", result.keys())
        print("Audio file transcribed in {:.2f} seconds".format(time.time() - start_time))

        output_file_path = os.path.splitext(audio_file_path)[0] + "_transcribed.txt"
        output_file_path_2 = os.path.splitext(audio_file_path)[0] + "_transcribed_2.txt"
        print("Writing transcription to file:", output_file_path)
        with open(output_file_path_2, "w") as f:
            f.write(str(result))  # convert dictionary to string before writing to file
        start_time = time.time()

        sentences = []
        current_sentence = ''
        prev_end_time = None

        for seg in result['segments']:
            seg_text = seg['text'].strip()

            if current_sentence:
                current_sentence += ' '

            if not prev_end_time:
                prev_end_time = seg['start']

            # If the segment text starts in the middle of a sentence
            if seg_text[0].islower():
                current_sentence += seg_text
            else:
                # Add the previous sentence to the sentences list
                if current_sentence:
                    sentences.append((current_sentence.strip(), prev_end_time, seg['start']))

                # Start a new sentence
                current_sentence = seg_text

            # If the segment text ends in the middle of a sentence
            if seg_text[-1] not in ['.', '?', '!']:
                continue

            # Add the current sentence to the sentences list
            sentence_duration = seg['end'] - prev_end_time
            if sentence_duration < 2 and sentences:
                # Merge with the previous sentence if its duration is less than 2 seconds
                prev_sentence, _, prev_end_time = sentences.pop()
                merged_sentence = prev_sentence + ' ' + current_sentence
                sentences.append((merged_sentence.strip(), prev_end_time, seg['end']))
            else:
                sentences.append((current_sentence.strip(), seg['start'], seg['end']))
                prev_end_time = seg['end']

            # Reset current sentence
            current_sentence = ''

        # Write the sentences to the output file
        with open(output_file_path, "w") as f:
            for sentence, start, end in sentences:
                f.write(f"{sentence}\t{start:.2f}\t{end:.2f}\n")

        print("Transcription written to file in {:.2f} seconds".format(time.time() - start_time))
        

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