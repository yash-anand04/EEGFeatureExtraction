import asyncio
import time
import os
import csv 
from datetime import datetime
from pylsl import StreamInlet, resolve_streams
import keyboard
import nest_asyncio

nest_asyncio.apply()

# ==============================
# Configuration
# ==============================
TRIAL_COUNT = 20
TRIAL_START = 1
RECORD_DURATION = 30        # seconds per trial 
REST_BETWEEN_TRIALS = 0     # rest duration between trials

# Output directory on desktop
BASE_FOLDER = os.path.join(r'D:\Projects\PE sem 6 (Reconstruction and feature extraction)\Data\Baseline (with_audio_and_visual_stimulus)')

# ==============================
# Utility Functions
# ==============================

# Create the base folder if it doesn't exist
def ensure_folder():
    if not os.path.exists(BASE_FOLDER):
        os.makedirs(BASE_FOLDER)
        print(f"📁 Created base folder at: {BASE_FOLDER}")

# ==============================
# Main function
# ==============================
async def main():
    ensure_folder()

    print("\n🔑 Press SPACE to start data collection...")
    keyboard.wait('space')

    # Wait 10 seconds after space is pressed before starting collection
    wait_seconds = 10
    print(f"⏳ Waiting {wait_seconds} seconds before starting data collection...")
    for remaining in range(wait_seconds, 0, -1):
        print(f"Starting in {remaining}...", end='\r')
        await asyncio.sleep(1)
    print("Starting now.            ")

    # =====================
    # EEG stream connection
    # =====================
    print("🔍 Searching for EEG stream...")
    lsl_streams = resolve_streams()
    inlet = None
    for stream in lsl_streams:
        if 'EEG' in stream.name() or 'EEG' in stream.type():
            inlet = StreamInlet(stream)
            print(f"✅ EEG stream found: {stream.name()} ({stream.type()})")
            break
    if not inlet:
        print("❌ No EEG stream found. Exiting.")
        return

    # =====================
    # Trial loop
    # =====================
    for trial in range(TRIAL_START,TRIAL_COUNT + TRIAL_START):
        print(f"\n🎬 Starting Trial {trial}/{TRIAL_COUNT} — Recording for {RECORD_DURATION} seconds...")

        # Get EEG channel count
        sample, _ = inlet.pull_sample()
        eeg_channel_count = len(sample)

        # Prepare header and data rows
        header = ['timestamp'] + [f'eeg{i+1}' for i in range(eeg_channel_count)]
        data_rows = []

        num_samples = int(RECORD_DURATION * 100)
        interval = 0.01  # 10ms

        for n in range(num_samples):
            tick_time = datetime.now()
            eeg_sample, _ = inlet.pull_sample(timeout=0.005)
            if eeg_sample:
                eeg_values = list(eeg_sample)
            else:
                eeg_values = [0] * eeg_channel_count

            row = [tick_time.strftime('%H:%M:%S.%f')] + eeg_values
            data_rows.append(row)

            await asyncio.sleep(interval)

        # Save EEG data
        trial_folder = os.path.join(BASE_FOLDER, f"trial_{trial:02d}")
        os.makedirs(trial_folder, exist_ok=True)
        filepath = os.path.join(trial_folder, "eeg_data.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        print(f"✅ Saved: {filepath}")

        if trial < TRIAL_COUNT:
            print(f"🛌 Resting for {REST_BETWEEN_TRIALS} seconds before next trial...")
            await asyncio.sleep(REST_BETWEEN_TRIALS)

    print("\n✅ All trials completed successfully.")

# ==============================
# Run the script
# ==============================
if __name__ == '__main__':
    asyncio.run(main())
