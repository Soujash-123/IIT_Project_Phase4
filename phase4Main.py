import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
import speech_recognition as sr
import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import threading

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from moviepy.editor import ImageSequenceClip, AudioFileClip

# Load the saved model and label encoder
model = tf.keras.models.load_model('audio_classification_model.keras')
labelencoder = joblib.load('labelencoder.pkl')


# Function to extract features
def feature_extraction(audio, sample_rate):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# Function to make predictions
def predict_segment(audio_segment, sample_rate):
    features = feature_extraction(audio_segment, sample_rate)
    features = features.reshape(1, -1)  # Reshape for model input
    prediction = model.predict(features)
    predicted_label = labelencoder.inverse_transform(np.argmax(prediction, axis=1))[0]
    return predicted_label


# Highlight the segments with specific labels
threatening_labels = ["gun_shot", "dog_bark", "siren"]


# Load and display the original audio with highlighted predictions
def display_waveform_with_highlights(file, output_video='output_video.mp4', segment_duration=1.0):
    audio, sample_rate = librosa.load(file)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio, sr=sample_rate)

    # Segment the audio and predict labels for each segment
    total_duration = len(audio) / sample_rate
    segment_samples = int(segment_duration * sample_rate)
    num_segments = int(total_duration / segment_duration)

    frames = []

    for i in range(num_segments):
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples
        segment = audio[start_sample:end_sample]
        predicted_label = predict_segment(segment, sample_rate)

        # Highlight the segment if it matches the threatening labels
        color = 'red' if predicted_label in threatening_labels else 'green'
        plt.axvspan(start_sample / sample_rate, end_sample / sample_rate, color=color, alpha=0.5)

        # Save the current frame
        frame_path = f'frame_{i:05d}.png'
        plt.savefig(frame_path)
        frames.append(frame_path)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], color='red', lw=4, label='Threatening'),
                       plt.Line2D([0], [0], color='green', lw=4, label='Non-Threatening')]
    plt.legend(handles=legend_elements)

    plt.title('Waveform with Highlighted Predictions')

    # Close the plot to release memory
    plt.close()

    # Create video from frames
    clip = ImageSequenceClip(frames, fps=1/segment_duration)
    audio_clip = AudioFileClip(file)
    video = clip.set_audio(audio_clip)
    video.write_videofile(output_video, codec='libx264')

    # Clean up the frames
    for frame in frames:
        os.remove(frame)


# Function to extract MFCC features from a segment of an audio file
def extract_mfcc(segment, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to split audio into fixed-length overlapping segments
def segment_audio(y, sr, segment_length=2, overlap=1.5):
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)
    segments = []
    start = 0

    while start + segment_samples <= len(y):
        segments.append(y[start:start + segment_samples])
        start += segment_samples - overlap_samples

    # Include last segment if not already included
    if start < len(y):
        segments.append(y[start:])

    return segments

# Function to map detailed emotions to broader categories
def map_emotion_to_category(emotion):
    negative_emotions = ['angry', 'disgust', 'fear']
    positive_emotions = ['happy', 'surprised']
    neutral_emotions = ['sad', 'neutral']

    if emotion in negative_emotions:
        return 'negative'
    elif emotion in positive_emotions:
        return 'positive'
    elif emotion in neutral_emotions:
        return 'neutral'
    return 'unknown'

# Function to enforce no two consecutive segments with the same emotion category
def enforce_emotion_consistency(emotion_categories):
    for i in range(1, len(emotion_categories)):
        if emotion_categories[i] == emotion_categories[i - 1]:
            # Change current segment emotion to 'neutral' if it matches previous
            emotion_categories[i] = 'neutral'
    return emotion_categories

# Function to plot waveplot with colored segments for specific emotions
def plot_colored_waveplot(y, sr, segments, emotion_labels, segment_length, overlap):
    plt.figure(figsize=(10, 4))
    plt.title("Waveplot with Emotion Highlights", size=20)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)

    color_map = {
        'Threat': 'red',
        'Probable Suspicion': 'orange',
        'Not a Threat': 'green',
        'neutral': 'yellow'
    }

    legend_labels = set()
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)
    start = 0

    for i, label in enumerate(emotion_labels):
        end = start + segment_samples
        color = color_map.get(label, 'gray')  # Use gray for any unexpected labels
        if label not in legend_labels:
            plt.axvspan(start / sr, end / sr, color=color, alpha=0.5, label=label)
            legend_labels.add(label)
        else:
            plt.axvspan(start / sr, end / sr, color=color, alpha=0.5)
        start += segment_samples - overlap_samples

    plt.legend()
    plt.show()

# Function to analyze audio emotions
def analyze_audio_emotions(audio_path, model, segment_length=2, overlap=1.5):
    # Load the test audio file
    y, sr = librosa.load(audio_path)

    # Segment the audio file
    segments = segment_audio(y, sr, segment_length, overlap)

    # Classify each segment
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    emotion_results = []
    emotion_categories = []

    for segment in segments:
        # Extract MFCC features for the segment
        segment_features = extract_mfcc(segment, sr)
        segment_features = np.expand_dims(segment_features, axis=0)
        segment_features = np.expand_dims(segment_features, axis=-1)

        # Predict emotion label for the segment
        predicted_probabilities = model.predict(segment_features)
        predicted_emotion_index = np.argmax(predicted_probabilities)
        predicted_emotion_label = emotions[predicted_emotion_index]

        # Map to broader category
        emotion_category = map_emotion_to_category(predicted_emotion_label)

        # Store the result
        emotion_results.append(predicted_emotion_label)
        emotion_categories.append(emotion_category)

    # Enforce no two consecutive segments with the same emotion category
    emotion_categories = enforce_emotion_consistency(emotion_categories)

    # Return the results for further processing
    return emotion_categories, y, sr, segments

# Function to recognize speech from wav file
def recognize_speech_from_wav(wav_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
        try:
            user_input = recognizer.recognize_google(audio)
            return user_input
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            return ""
        except sr.RequestError as e:
            print("Error occurred; {0}".format(e))
            return ""

# Function to predict label from recognized speech using Roberta model
def predict_label(user_input, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        predicted_label_id = torch.argmax(logits, dim=1).item()
    return predicted_label_id, logits

# Function to run the first thread (code1)
def run_code1(audio_path, model, result_dict):
    emotion_categories, y, sr, segments = analyze_audio_emotions(
        audio_path=audio_path,
        model=model,
        segment_length=2,
        overlap=1.5
    )
    result_dict['emotion_categories'] = emotion_categories
    result_dict['y'] = y
    result_dict['sr'] = sr
    result_dict['segments'] = segments

# Function to run the second thread (code2)
def run_code2(wav_file, tokenizer, model, result_dict):
    user_input = recognize_speech_from_wav(wav_file)
    if user_input:
        label, _ = predict_label(user_input, tokenizer, model)
        result_dict['label_code2'] = label

# Main function to combine the results from both threads
def main(audio_path):
    model_path_code1 = 'speech_emotion_recognition_model.keras'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(current_dir, 'saved_model3')

    # Load models for both codes
    model_code1 = load_model(model_path_code1)
    tokenizer_code2 = RobertaTokenizer.from_pretrained(save_directory)
    model_code2 = RobertaForSequenceClassification.from_pretrained(save_directory)

    # Dictionary to store results from threads
    result_dict = {}

    # Create threads for running code1 and code2
    thread1 = threading.Thread(target=run_code1, args=(audio_path, model_code1, result_dict))
    thread2 = threading.Thread(target=run_code2, args=(audio_path, tokenizer_code2, model_code2, result_dict))

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread1.join()
    thread2.join()

    # Retrieve results from both threads
    emotion_categories = result_dict.get('emotion_categories', [])
    y = result_dict.get('y')
    sr = result_dict.get('sr')
    segments = result_dict.get('segments')
    label_code2 = result_dict.get('label_code2')

    # Determine the final label based on the given specifications
    final_label = []
    for emotion in emotion_categories:
        if emotion == 'negative' and label_code2 in [0, 1, 2]:
            if label_code2 == 0 or label_code2 == 1:
                final_label.append('Threat')
            elif label_code2 == 2:
                final_label.append('Probable Suspicion')
        elif emotion == 'positive' and label_code2 in [0, 1, 2]:
            if label_code2 == 0 or label_code2 == 2:
                final_label.append('Not a Threat')
            elif label_code2 == 1:
                final_label.append('Probable Suspicion')
        elif emotion == 'neutral':
            final_label.append(emotion)

    # Print the final labels
    print("Final Labels: ", final_label)

    # Plot the final waveplot with colored segments for specific emotions
    if y is not None and sr is not None and segments is not None:
        plot_colored_waveplot(y, sr, segments, final_label, 5, 0.125)

if __name__ == "__main__":
    audio_path = "output.wav"

    # Dictionary to store results from both main and display_waveform_with_highlights
    result_dict_main = {}

    # Create threads for running main and display_waveform_with_highlights
    thread_main = threading.Thread(target=main, args=(audio_path,))
    thread_highlight = threading.Thread(target=display_waveform_with_highlights, args=(audio_path,))

    # Start the threads
    thread_main.start()
    thread_highlight.start()

    # Wait for both threads to complete
    thread_main.join()
    thread_highlight.join()
