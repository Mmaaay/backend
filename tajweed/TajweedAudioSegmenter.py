import numpy as np
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler
import librosa
import scipy.signal
import json
from typing import Dict, List, Tuple

class TajweedAudioSegmenter:
    def __init__(self, audio_file_path: str, tajweed_output: Dict):
        """
        Initialize the TajweedAudioSegmenter with the audio file and tajweed output from process_audio_with_tajweed.

        Args:
            audio_file_path (str): Path to the input audio file.
            tajweed_output (Dict): Output from process_audio_with_tajweed containing transcription, words, char_tajweed, and char_timestamps.
        """
        self.audio_file_path = audio_file_path
        self.tajweed_output = tajweed_output
        self.audio, self.sr = librosa.load(audio_file_path, sr=None, mono=True)
        self.frame_size = 1024
        self.hop_size = self.frame_size // 2
        # Placeholder for reference features (replace with actual data)
        self.reference_mfccs = None  # Load from file or database if available
        self.reference_pitch = None  # Load from file or database if available

    def _extract_words_and_timestamps(self) -> List[Dict]:
        """
        Extract words, timestamps, and associated tajweed rules from the tajweed_output.

        Returns:
            List[Dict]: List of dictionaries containing word, start_time, end_time, and tajweed_rules.
        """
        # Directly use the 'words' field from tajweed_output (from process_audio_with_tajweed)
        words_data = self.tajweed_output.get('words', [])
        if not words_data:
            print("Warning: No words found in tajweed_output.")
            return []

        # Ensure the words_data format matches expectations
        formatted_words = []
        for word_info in words_data:
            formatted_words.append({
                'word': word_info.get('word', ''),
                'start_time': word_info.get('start_time', 0.0),
                'end_time': word_info.get('end_time', 0.0),
                'tajweed_rules': word_info.get('tajweed_rules', [])
            })

        return formatted_words

    def _segment_audio(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Segment the audio based on start and end times.

        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.

        Returns:
            np.ndarray: Audio segment.
        """
        start_sample = int(start_time * self.sr)
        end_sample = min(int(end_time * self.sr), len(self.audio))
        if start_sample < 0 or end_sample > len(self.audio):
            print(f"Error: Sample range out of bounds for start: {start_time}s, end: {end_time}s")
            return None
        return self.audio[start_sample:end_sample]

    def _high_pass_filter(self, audio: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """
        Apply a high-pass filter to the audio.

        Args:
            audio (np.ndarray): Input audio.
            cutoff (float): Cutoff frequency in Hz.

        Returns:
            np.ndarray: Filtered audio.
        """
        sos = scipy.signal.butter(4, cutoff, btype='high', fs=self.sr, output='sos')
        return scipy.signal.sosfilt(sos, audio)

    def _extract_features(self, audio_segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract MFCCs, pitch, RMS, and ZCR features from an audio segment using librosa.

        Args:
            audio_segment (np.ndarray): Audio segment to analyze.

        Returns:
            Tuple containing MFCCs, pitch values, RMS values, and ZCR values.
        """
        # High-pass filter
        y_filtered = self._high_pass_filter(audio_segment, cutoff=80)

        # Pitch extraction
        f0, voiced_flag, _ = librosa.pyin(
            y_filtered,
            fmin=80,
            fmax=1000,
            sr=self.sr,
            frame_length=self.frame_size,
            hop_length=self.hop_size,
            fill_na=0.0
        )
        pitch_values = f0[voiced_flag]  # Only keep voiced frames

        # RMS
        rms_values = librosa.feature.rms(
            y=y_filtered,
            frame_length=self.frame_size,
            hop_length=self.hop_size
        )[0]

        # ZCR
        zcr_values = librosa.feature.zero_crossing_rate(
            y=y_filtered,
            frame_length=self.frame_size,
            hop_length=self.hop_size
        )[0]

        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=y_filtered,
            sr=self.sr,
            n_mfcc=13,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            window='hann'
        ).T  # Transpose to match expected format (frames x coefficients)

        return mfccs, pitch_values, rms_values, zcr_values

    def _detect_madd(self, rms_values: np.ndarray, zcr_values: np.ndarray) -> bool:
        """
        Detect madd based on RMS and ZCR thresholds.

        Args:
            rms_values (np.ndarray): RMS values for the audio segment.
            zcr_values (np.ndarray): ZCR values for the audio segment.

        Returns:
            bool: True if madd is detected, False otherwise.
        """
        # Note: Thresholds may need recalibration due to librosa's scaling
        rms_threshold = 0.001
        zcr_threshold = 0.03
        elongation_frames = [
            i for i in range(len(rms_values))
            if rms_values[i] > rms_threshold and zcr_values[i] < zcr_threshold
        ]
        return len(elongation_frames) > len(zcr_values) * 0.40

    def _compare_features(self, user_mfccs: np.ndarray, user_pitch: np.ndarray) -> Tuple[float, float]:
        """
        Compare user MFCCs and pitch with reference features using DTW.

        Args:
            user_mfccs (np.ndarray): User's MFCC features.
            user_pitch (np.ndarray): User's pitch values.

        Returns:
            Tuple[float, float]: Similarity scores for MFCCs and pitch.
        """
        if self.reference_mfccs is None or self.reference_pitch is None:
            print("Warning: Reference features not provided. Skipping comparison.")
            return 0.0, 0.0

        # Normalize pitch
        user_pitch_norm = (user_pitch - np.mean(user_pitch)) / np.std(user_pitch) if np.std(user_pitch) != 0 else user_pitch
        ref_pitch_norm = (self.reference_pitch - np.mean(self.reference_pitch)) / np.std(self.reference_pitch) if np.std(self.reference_pitch) != 0 else self.reference_pitch
        distance, _ = fastdtw(user_pitch_norm, ref_pitch_norm, dist=lambda x, y: abs(x - y))
        max_length = max(len(user_pitch_norm), len(ref_pitch_norm))
        max_possible_difference = max(
            abs(np.max(user_pitch_norm) - np.min(ref_pitch_norm)),
            abs(np.max(ref_pitch_norm) - np.min(user_pitch_norm))
        )
        max_distance = max_length * max_possible_difference
        pitch_similarity = (1 - distance / max_distance) * 100 if max_distance != 0 else 100.0

        # Normalize MFCCs
        scaler = StandardScaler()
        user_mfccs_norm = scaler.fit_transform(user_mfccs)
        ref_mfccs_norm = scaler.fit_transform(self.reference_mfccs)
        D, _ = librosa.sequence.dtw(user_mfccs_norm.T, ref_mfccs_norm.T, metric='euclidean')
        distance = D[-1, -1]
        num_frames = max(user_mfccs_norm.shape[0], ref_mfccs_norm.shape[0])
        max_possible_difference = max(
            np.max(np.linalg.norm(user_mfccs_norm - np.min(ref_mfccs_norm, axis=0), axis=1)),
            np.max(np.linalg.norm(ref_mfccs_norm - np.min(user_mfccs_norm, axis=0), axis=1))
        )
        max_distance = num_frames * max_possible_difference
        mfcc_similarity = (1 - distance / max_distance) * 100 if max_distance != 0 else 100.0

        return mfcc_similarity, pitch_similarity

    def analyze_tajweed(self) -> Dict:
        """
        Segment the audio and analyze tajweed rules for each word.

        Returns:
            Dict: Analysis results for each word, including madd detection and feature comparison.
        """
        words_data = self._extract_words_and_timestamps()
        results = []

        for word_data in words_data:
            word = word_data['word']
            start_time = word_data['start_time']
            end_time = word_data['end_time']
            tajweed_rules = word_data['tajweed_rules']

            # Skip words without tajweed rules for now (modify if other rules need analysis)
            if not tajweed_rules:
                results.append({
                    'word': word,
                    'start_time': start_time,
                    'end_time': end_time,
                    'tajweed_rules': [],
                    'madd_detected': False,
                    'mfcc_similarity': None,
                    'pitch_similarity': None
                })
                continue

            # Segment audio
            audio_segment = self._segment_audio(start_time, end_time)
            if audio_segment is None:
                results.append({
                    'word': word,
                    'start_time': start_time,
                    'end_time': end_time,
                    'tajweed_rules': tajweed_rules,
                    'madd_detected': False,
                    'mfcc_similarity': None,
                    'pitch_similarity': None,
                    'error': 'Failed to segment audio'
                })
                continue

            # Extract features
            mfccs, pitch_values, rms_values, zcr_values = self._extract_features(audio_segment)

            # Detect madd
            madd_detected = self._detect_madd(rms_values, zcr_values)

            # Compare features with reference
            mfcc_similarity, pitch_similarity = self._compare_features(mfccs, pitch_values)

            results.append({
                'word': word,
                'start_time': start_time,
                'end_time': end_time,
                'tajweed_rules': tajweed_rules,
                'madd_detected': madd_detected,
                'mfcc_similarity': mfcc_similarity,
                'pitch_similarity': pitch_similarity
            })

        return {'words_analysis': results}