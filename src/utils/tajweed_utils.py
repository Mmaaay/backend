import os
import io
import time
import torch
import json
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from .quran_tajweed.tajweed_classifier import label_ayah
from .quran_tajweed.tree import Exemplar, json2tree

# All utility functions from main.py that are not endpoints

def load_rule_trees():
    """Load tajweed rule trees from JSON files"""
    rule_trees = {}

    # Try several possible locations for the rule trees
    # Add more robust search paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "quran_tajweed", "rule_trees"),
        os.path.join(base_dir, "..", "quran_tajweed", "rule_trees"),
        os.path.join(base_dir, "..", "..", "quran_tajweed", "rule_trees"),
        "quran_tajweed/rule_trees",
        "./quran_tajweed/rule_trees",
        "../quran_tajweed/rule_trees",
        "/home/mazen/coding/Quran-back/backend/tajweed/quran_tajweed/rule_trees",
    ]

    rule_start_files = []
    found_path = None

    for path in possible_paths:
        pattern = os.path.join(path, "*.start.json")
        files = glob.glob(pattern)
        if files:
            rule_start_files = files
            found_path = path
            print(f"Found rule trees in: {os.path.abspath(path)}")
            break

    if not rule_start_files:
        print(
            "Warning: No tajweed rule files found. Tajweed classification will not work."
        )
        print(
            "Please ensure the rule_trees directory exists with .start.json and .end.json files."
        )
        print("Searched paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        return {}

    # Load each rule
    for start_file in rule_start_files:
        rule_name = os.path.basename(start_file).partition(".")[0]
        end_file = start_file.replace(".start.", ".end.")

        try:
            with open(start_file, "r", encoding="utf-8") as f_start, open(
                end_file, "r", encoding="utf-8"
            ) as f_end:
                rule_trees[rule_name] = {
                    "start": json2tree(json.load(f_start)),
                    "end": json2tree(json.load(f_end)),
                }
            print(f"Loaded rule: {rule_name}")
        except Exception as e:
            print(f"Error loading rule {rule_name}: {str(e)}")

    return rule_trees

    # ...existing code from main.py...
    pass

def transcribe_with_timestamps(asr_model, audio_file_path):
    """Transcribe audio and return word and character timestamps"""
    print(f"Transcribing {audio_file_path}...")
    hypotheses = asr_model.transcribe(
        [audio_file_path], channel_selector=0, timestamps=True
    )

    if not hypotheses:
        print("Error: Failed to get transcription.")
        return None, None, None

    transcription = hypotheses[0].text

    # Check if timestamp data is available
    if not hasattr(hypotheses[0], "timestamp") or "word" not in hypotheses[0].timestamp:
        print(f"Error: No word timestamps available in transcription.{hypotheses[0]}")
        return transcription, None, None

    word_timestamps = hypotheses[0].timestamp["word"]
    char_timestamps = None

    # Check if character timestamps are available
    if "char" in hypotheses[0].timestamp:
        char_timestamps = hypotheses[0].timestamp["char"]
        # Log the structure of char_timestamps for debugging
        if char_timestamps and len(char_timestamps) > 0:
            print(f"Character timestamp example format: {char_timestamps[0]}")
        print(f"Found {len(char_timestamps)} characters with timestamps")

    print(f"Transcription: {transcription}")
    print(f"Found {len(word_timestamps)} words with timestamps")

    return transcription, word_timestamps, char_timestamps

def match_tajweed_to_char_timestamps(annotations, char_timestamps, text):
    """Match tajweed annotations with character timestamps"""
    # Create a mapping from text position to timestamp
    char_time_map = {}

    # Process character timestamps based on available format
    for i, char_info in enumerate(char_timestamps):
        # Handle different character timestamp formats
        if isinstance(char_info, dict):
            # Format 1: Dictionary with 'idx' key
            if "idx" in char_info:
                char_idx = char_info["idx"]
                start_time = char_info.get("start", 0)
                end_time = char_info.get("end", 0)
            # Format 2: Dictionary with character index and time
            elif "char" in char_info:
                char_idx = i  # Use position in the list as index
                start_time = char_info.get("start_time", 0)
                end_time = char_info.get("end_time", 0)
            # Format 3: NeMo format with different keys
            else:
                # Try to extract common timestamp fields
                char_idx = i
                start_time = char_info.get("start", char_info.get("begin_time", 0))
                end_time = char_info.get("end", char_info.get("end_time", 0))
        # Format 4: Timestamps as arrays with [char_idx, start, end]
        elif isinstance(char_info, (list, tuple)) and len(char_info) >= 3:
            char_idx = char_info[0] if isinstance(char_info[0], int) else i
            start_time = char_info[1]
            end_time = char_info[2]
        # Format 5: Use position in array as character index
        else:
            char_idx = i
            start_time = 0
            end_time = 0
            # Try to extract time if it's a numeric value
            if isinstance(char_info, (int, float)):
                end_time = float(char_info)
                start_time = (
                    end_time - 0.1
                )  # Assume 100ms duration if only one time is given

        # Map character index to its timestamp
        if char_idx < len(text):
            char_time_map[char_idx] = {
                "char": text[char_idx] if char_idx < len(text) else "",
                "start_time": start_time,
                "end_time": end_time,
            }

    # If we couldn't extract timestamps with above methods, use a linear mapping approach
    if not char_time_map and char_timestamps:
        total_chars = len(text)
        total_duration = (
            max(w["end"] for w in char_timestamps)
            if isinstance(char_timestamps[0], dict) and "end" in char_timestamps[0]
            else 10.0
        )

        # Create a simple linear mapping of characters to time
        for i in range(total_chars):
            rel_pos = i / total_chars
            char_time_map[i] = {
                "char": text[i],
                "start_time": rel_pos * total_duration,
                "end_time": (rel_pos + 1 / total_chars) * total_duration,
            }

    # Match annotations with precise character timestamps
    tajweed_chars = []

    for annotation in annotations:
        rule_name = annotation["rule"]
        start_idx = annotation["start"]
        end_idx = annotation["end"]

        # Find start and end timestamps using character indices
        start_time = None
        end_time = None

        # Look for exact or nearby character positions
        for offset in range(5):  # Try nearby positions if exact position isn't found
            if start_idx + offset in char_time_map:
                start_time = char_time_map[start_idx + offset]["start_time"]
                break
            if start_idx - offset in char_time_map and start_idx - offset >= 0:
                start_time = char_time_map[start_idx - offset]["start_time"]
                break

        for offset in range(5):
            if end_idx + offset in char_time_map:
                end_time = char_time_map[end_idx + offset]["end_time"]
                break
            if end_idx - offset in char_time_map and end_idx - offset >= 0:
                end_time = char_time_map[end_idx - offset]["end_time"]
                break

        # If we found valid timestamps, add the annotation
        if start_time is not None and end_time is not None:
            tajweed_chars.append(
                {
                    "rule": rule_name,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "text": text[start_idx : end_idx + 1]
                    if start_idx < len(text) and end_idx < len(text)
                    else "",
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

    return tajweed_chars

    # ...existing code from main.py...
    pass

def classify_tajweed_with_char_precision(
    text, word_timestamps, char_timestamps, rule_trees
):
    """Classify Tajweed rules with character-level precision"""
    if not rule_trees:
        print("Error: No rule trees provided for tajweed classification.")
        return [], []

    # Get all tajweed annotations
    surah_num = 1  # Placeholder for classification
    ayah_num = 7  # Placeholder for classification

    # Run the tajweed classifier
    result = label_ayah(
        (
            surah_num,
            ayah_num,
            "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ",
            rule_trees,
        )
    )
    print(f"Classified tajweed rules: {result}")
    annotations = result["annotations"]

    # Process character-level tajweed rules if character timestamps are available
    tajweed_chars = []
    if char_timestamps:
        tajweed_chars = match_tajweed_to_char_timestamps(
            annotations, char_timestamps, text
        )

    # Map character annotations to words
    words_with_tajweed = []

    # First, build a map of character ranges for each word
    word_char_ranges = []
    char_index = 0

    for word_info in word_timestamps:
        word = word_info["word"]
        word_start = char_index

        # Find the word in the text, starting from the current position
        while char_index < len(text):
            # Skip whitespace
            while char_index < len(text) and text[char_index].isspace():
                char_index += 1

            # Check if we found the word
            word_end = char_index + len(word)
            if (
                word_end <= len(text)
                and text[char_index:word_end].lower() == word.lower()
            ):
                word_char_ranges.append(
                    {
                        "word": word,
                        "start_idx": char_index,
                        "end_idx": word_end - 1,
                        "start_time": word_info["start"],
                        "end_time": word_info["end"],
                    }
                )
                char_index = word_end
                break

            # If not, try to find it with more flexibility
            found = False
            for i in range(max(0, char_index - 5), min(len(text), char_index + 20)):
                if (
                    i + len(word) <= len(text)
                    and text[i : i + len(word)].lower() == word.lower()
                ):
                    word_char_ranges.append(
                        {
                            "word": word,
                            "start_idx": i,
                            "end_idx": i + len(word) - 1,
                            "start_time": word_info["start"],
                            "end_time": word_info["end"],
                        }
                    )
                    char_index = i + len(word)
                    found = True
                    break

            if found:
                break

            # If still not found, just use the word as is and advance
            print(
                f"Warning: Could not find word '{word}' in text at position {char_index}"
            )
            word_char_ranges.append(
                {
                    "word": word,
                    "start_idx": -1,  # Mark as not found in text
                    "end_idx": -1,
                    "start_time": word_info["start"],
                    "end_time": word_info["end"],
                }
            )
            char_index += len(word)
            break

    # Now, map tajweed rules to words using character indices
    for word_range in word_char_ranges:
        word = word_range["word"]
        start_time = word_range["start_time"]
        end_time = word_range["end_time"]
        start_idx = word_range["start_idx"]
        end_idx = word_range["end_idx"]

        word_rules = []

        # Skip words not found in text
        if start_idx == -1:
            words_with_tajweed.append(
                {
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time,
                    "tajweed_rules": [],
                }
            )
            continue

        # Check each annotation to see if it overlaps with this word
        for annotation in annotations:
            rule_start = annotation["start"]
            rule_end = annotation["end"]

            # Check if the rule overlaps with this word
            if rule_start <= end_idx and rule_end >= start_idx:
                # Calculate relative positions within the word
                rel_start = max(rule_start, start_idx)
                rel_end = min(rule_end, end_idx)

                # Convert character positions to time
                if char_timestamps:
                    # Try to find exact character timestamps
                    rule_start_time = None
                    rule_end_time = None

                    # Create a mapping of character positions to their timestamps
                    char_time_map = {}
                    for i, char_info in enumerate(char_timestamps):
                        if isinstance(char_info, dict):
                            # Extract the character index from the dictionary using various possible keys
                            char_idx = None
                            if "idx" in char_info:
                                char_idx = char_info["idx"]
                            elif "index" in char_info:
                                char_idx = char_info["index"]
                            else:
                                char_idx = (
                                    i  # Use position as index if no explicit index
                                )

                            # Extract timing information
                            start = char_info.get(
                                "start", char_info.get("begin_time", 0)
                            )
                            end = char_info.get("end", char_info.get("end_time", 0))

                            # Store in our map if valid
                            if char_idx is not None and char_idx < len(text):
                                char_time_map[char_idx] = {"start": start, "end": end}
                        else:
                            # For non-dictionary formats, use position as index
                            char_time_map[i] = {
                                "start": float(char_info)
                                if isinstance(char_info, (int, float))
                                else i / len(text) * 10,
                                "end": float(char_info) + 0.1
                                if isinstance(char_info, (int, float))
                                else (i + 1) / len(text) * 10,
                            }

                    # Look for the start time
                    for offset in range(5):
                        pos = rel_start + offset
                        if pos in char_time_map:
                            rule_start_time = char_time_map[pos]["start"]
                            break
                        pos = rel_start - offset
                        if pos >= 0 and pos in char_time_map:
                            rule_start_time = char_time_map[pos]["start"]
                            break

                    # Look for the end time
                    for offset in range(5):
                        pos = rel_end + offset
                        if pos in char_time_map:
                            rule_end_time = char_time_map[pos]["end"]
                            break
                        pos = rel_end - offset
                        if pos >= 0 and pos in char_time_map:
                            rule_end_time = char_time_map[pos]["end"]
                            break

                    # If we couldn't find precise timestamps, interpolate
                    if rule_start_time is None:
                        rel_pos_start = (rel_start - start_idx) / max(
                            1, end_idx - start_idx + 1
                        )
                        rule_start_time = (
                            start_time + (end_time - start_time) * rel_pos_start
                        )

                    if rule_end_time is None:
                        rel_pos_end = (rel_end - start_idx) / max(
                            1, end_idx - start_idx + 1
                        )
                        rule_end_time = (
                            start_time + (end_time - start_time) * rel_pos_end
                        )
                else:
                    # Fallback to relative position within the word
                    rel_pos_start = (rel_start - start_idx) / max(
                        1, end_idx - start_idx + 1
                    )
                    rel_pos_end = (rel_end - start_idx) / max(
                        1, end_idx - start_idx + 1
                    )
                    rule_start_time = (
                        start_time + (end_time - start_time) * rel_pos_start
                    )
                    rule_end_time = start_time + (end_time - start_time) * rel_pos_end

                # Add the rule to this word
                word_rules.append(
                    {
                        "rule": annotation["rule"],
                        "start_idx": rel_start,
                        "end_idx": rel_end,
                        "text": text[rel_start : rel_end + 1]
                        if rel_start < len(text) and rel_end < len(text)
                        else "",
                        "start_time": rule_start_time,
                        "end_time": rule_end_time,
                        "spans_words": rule_start < start_idx or rule_end > end_idx,
                    }
                )

        words_with_tajweed.append(
            {
                "word": word,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_time,
                "end_time": end_time,
                "tajweed_rules": word_rules,
            }
        )

    return words_with_tajweed, tajweed_chars

def process_audio_with_tajweed(audio_file_path, asr_model, rule_trees):
    """Complete pipeline to process audio, transcribe, and classify tajweed rules with timestamps"""

    # Transcribe audio with timestamps
    transcription, word_timestamps, char_timestamps = transcribe_with_timestamps(
        asr_model, audio_file_path
    )
    if not transcription or not word_timestamps:
        return None

    # Print debug information about character timestamps
    if char_timestamps and len(char_timestamps) > 0:
        print(f"Character timestamp format: {type(char_timestamps)}")
        print(f"First character timestamp: {char_timestamps[0]}")

    # Classify tajweed rules and align with timestamps
    words_with_tajweed, tajweed_chars = classify_tajweed_with_char_precision(
        transcription, word_timestamps, char_timestamps, rule_trees
    )

    # Create a safe representation of character timestamps for JSON export
    if char_timestamps:
        if isinstance(char_timestamps[0], dict):
            # Try to normalize the dictionary format
            safe_char_timestamps = []
            for i, c in enumerate(char_timestamps):
                if i < len(transcription):
                    safe_char = {
                        "index": i,
                        "char": transcription[i] if i < len(transcription) else "",
                        "start": c.get("start", c.get("begin_time", 0)),
                        "end": c.get("end", c.get("end_time", 0)),
                    }
                    safe_char_timestamps.append(safe_char)
        else:
            # Create a list of dictionaries with the character timestamps
            safe_char_timestamps = []
            for i, c in enumerate(char_timestamps):
                if i < len(transcription):
                    # Handle different possible formats
                    if isinstance(c, (list, tuple)) and len(c) >= 2:
                        char_dict = {
                            "index": i,
                            "char": transcription[i],
                            "start": c[0],
                            "end": c[1],
                        }
                    else:
                        # Use position as time estimate
                        char_dict = {
                            "index": i,
                            "char": transcription[i],
                            "start": float(c)
                            if isinstance(c, (int, float))
                            else i / len(transcription) * 10,
                            "end": float(c) + 0.1
                            if isinstance(c, (int, float))
                            else (i + 1) / len(transcription) * 10,
                        }
                    safe_char_timestamps.append(char_dict)
    else:
        safe_char_timestamps = []

    return {
        "transcription": transcription,
        "words": words_with_tajweed,
        "char_tajweed": tajweed_chars,
        "char_timestamps": safe_char_timestamps,
    }

def visualize_tajweed_results(result):
    """Visualize the tajweed classification results with timestamps"""
    if not result or "words" not in result:
        print("No results to visualize.")
        return

    # Define colors for different tajweed rules
    rule_colors = {
        "ghunnah": "red",
        "idghaam_ghunnah": "blue",
        "idghaam_no_ghunnah": "green",
        "ikhfa": "purple",
        "iqlab": "orange",
        "qalqalah": "brown",
        "madd_246": "pink",
        "madd_munfasil": "cyan",
        "madd_muttasil": "magenta",
        "lam_shamsiyyah": "yellow",
        # Add more rules and colors as needed
    }

    # Create a timeline visualization
    fig, ax = plt.subplots(figsize=(15, 5))

    # Calculate total duration
    words = result["words"]
    if not words:
        print("No words with timestamps found.")
        return

    total_duration = max(word["end_time"] for word in words)

    # Set up the plot
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, len(words) + 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks(range(1, len(words) + 1))
    ax.set_yticklabels([word["word"] for word in words])
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Plot words as segments
    for i, word in enumerate(words):
        word_rect = patches.Rectangle(
            (word["start_time"], i + 0.5),
            word["end_time"] - word["start_time"],
            0.8,
            linewidth=1,
            edgecolor="black",
            facecolor="lightgray",
            alpha=0.5,
        )
        ax.add_patch(word_rect)

        # Plot tajweed rules
        y_offset = 0.1
        for rule in word["tajweed_rules"]:
            rule_name = rule["rule"]
            color = rule_colors.get(rule_name, "gray")

            rule_rect = patches.Rectangle(
                (rule["start_time"], i + 0.5 - y_offset),
                rule["end_time"] - rule["start_time"],
                0.2,
                linewidth=1,
                edgecolor="black",
                facecolor=color,
                label=rule_name,
            )
            ax.add_patch(rule_rect)
            y_offset += 0.25  # Stack multiple rules vertically

    # Create a legend for tajweed rules
    handles = [
        patches.Patch(color=color, label=rule) for rule, color in rule_colors.items()
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Tajweed Rules with Timestamps")
    plt.tight_layout()
    # plt.show()

    # Print detailed results
    print("\nDetailed Tajweed Analysis:")
    for i, word in enumerate(words):
        print(
            f"\nWord {i+1}: {word['word']} ({word['start_time']:.2f}s - {word['end_time']:.2f}s)"
        )
        if word["tajweed_rules"]:
            for rule in word["tajweed_rules"]:
                print(
                    f"  - {rule['rule']} rule: {rule['start_time']:.2f}s - {rule['end_time']:.2f}s"
                )
                if "text" in rule:
                    print(f"    Text: {rule['text']}")
        else:
            print("  (No tajweed rules)")

    # If character-level tajweed is available, show that too
    if "char_tajweed" in result and result["char_tajweed"]:
        print("\nCharacter-level Tajweed Rules:")
        for i, rule in enumerate(result["char_tajweed"]):
            print(
                f"\nRule {i+1}: {rule['rule']} ({rule['start_time']:.2f}s - {rule['end_time']:.2f}s)"
            )
            print(f"  Text: {rule['text']}")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    buffer.seek(0)
    return buffer

