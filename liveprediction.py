import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
import pyaudio
from pathlib import Path
import time
import argparse
import wget
import os
from reprint import output
from helpers import Interpolator, ratio_to_db, dbFS, rangemap
from datetime import datetime

# thresholds
PREDICTION_THRES = 0.8  # confidence
DBLEVEL_THRES = -35  # dB

# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0

###########################
# Model download
###########################
def download_model(url, output):
    return wget.download(url, output)


###########################
# Check Microphone
###########################
print("=====")
print("1 / 2: Checking Microphones... ")
print("=====")

import microphones

desc, mics, indices = microphones.list_microphones()
if len(mics) == 0:
    print("Error: No microphone found.")
    exit()

#############
# Read Command Line Args
#############
MICROPHONE_INDEX = indices[0]
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mic", help="Select which microphone / input device to use")
args = parser.parse_args()
try:
    if args.mic:
        MICROPHONE_INDEX = int(args.mic)
        print("User selected mic: %d" % MICROPHONE_INDEX)
    else:
        mic_in = input("Select microphone [%d]: " % MICROPHONE_INDEX).strip()
        if mic_in != "":
            MICROPHONE_INDEX = int(mic_in)
except:
    print("Invalid microphone")
    exit()

# Find description that matches the mic index
mic_desc = ""
for k in range(len(indices)):
    i = indices[k]
    if i == MICROPHONE_INDEX:
        mic_desc = mics[k]
print("Using mic: %s" % mic_desc)

###########################
# Download model, if it doesn't exist
###########################
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "/home/rock/ubicoustics/models/example_model.hdf5"
print("=====")
print("2 / 2: Checking model... ")
print("=====")
model_filename = "/home/rock/ubicoustics/models/example_model.hdf5"
ubicoustics_model = Path(model_filename)
if not ubicoustics_model.is_file():
    print("Downloading example_model.hdf5 [867MB]: ")
    download_model(MODEL_URL, MODEL_PATH)

##############################
# Load Deep Learning Model
##############################
print("Using deep learning model: %s" % (model_filename))
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    set_session(session)
    model = load_model(model_filename)
context = ubicoustics.everything

label = dict()
for k in range(len(context)):
    label[k] = context[k]

##############################
# Setup Audio Callback
##############################
m = 0
audio_rms = 0
candidate = ("-", 0.0)

# Prediction Interpolators
interpolators = []
for k in range(31):
    interpolators.append(Interpolator())

# Audio Input Callback
def audio_samples(in_data, frame_count, time_info, status_flags):
    global session
    global interpolators
    global audio_rms
    global candidate
    global m
    np_wav = np.fromstring(in_data, dtype=np.int16) / 32768.0  # Convert to [-1.0, +1.0]

    # Compute RMS and convert to dB
    rms = np.sqrt(np.mean(np_wav ** 2))
    db = dbFS(rms)
    interp = interpolators[30]
    interp.animate(interp.end, db, 1.0)

    # Make Predictions
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    with session.graph.as_default():
        set_session(session)
        if x.shape[0] != 0:
            x = x.reshape(len(x), 96, 64, 1)
            pred = model.predict(x)
            predictions.append(pred)

        for prediction in predictions:
            m = np.argmax(prediction[0])
            candidate = (ubicoustics.to_human_labels[label[m]], prediction[0, m])
            num_classes = len(prediction[0])
            for k in range(num_classes):
                interp = interpolators[k]
                prev = interp.end
                interp.animate(prev, prediction[0, k], 1.0)
    return (in_data, pyaudio.paContinue)


##############################
# Main Execution
##############################
while 1:
    ##############################
    # Setup Audio
    ##############################
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_samples,
        input_device_index=MICROPHONE_INDEX,
    )

    ##############################
    # Start Non-Blocking Stream
    ##############################
    os.system("cls" if os.name == "nt" else "clear")
    print("# Live Prediction Using Microphone: %s" % (mic_desc))
    stream.start_stream()
    while stream.is_active():
        last_pred = "-"
        while True:
            time.sleep(1.0 / FPS)  # 60fps
            for k in range(30):
                interp = interpolators[k]
                val = interp.update()

            # dB Levels
            interp = interpolators[30]
            db = interp.update()

            # Final Prediction
            pred = "-"
            event, conf = candidate
            pred = event
            if conf > PREDICTION_THRES and db > DBLEVEL_THRES:
                if pred != last_pred:
                    timestamp = datetime.now().strftime("%D %H:%M:%S")
                    print(f"{timestamp} {pred.upper()}")
                    with open("/home/rock/ubicoustics/sound_detection.txt", "w") as detection:
                        detection.writelines(f"{m}")
                    last_pred = pred
            else:
                if pred != last_pred:
                    with open("/home/rock/ubicoustics/sound_detection.txt", "w") as detection:
                        detection.writelines("-")
                    last_pred = "-"
