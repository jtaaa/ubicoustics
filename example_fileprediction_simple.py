import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
from vggish_input import waveform_to_examples, wavfile_to_examples
import numpy as np
import vggish_params
from pathlib import Path
import ubicoustics
import wget

###########################
# Download model, if it doesn't exist
###########################
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
print("=====")
print("Checking model... ")
print("=====")
model_filename = "models/example_model.hdf5"
ubicoustics_model = Path(model_filename)
if not ubicoustics_model.is_file():
    print("Downloading example_model.hdf5 [867MB]: ")
    wget.download(MODEL_URL, MODEL_PATH)

###########################
# Load Model
###########################
context = ubicoustics.everything
context_mapping = ubicoustics.context_mapping
trained_model = model_filename
other = True
selected_file = "example.wav"
selected_context = "everything"

print("Using deep learning model: %s" % (trained_model))
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    set_session(session)
    model = load_model(model_filename)
context = context_mapping[selected_context]

label = dict()
for k in range(len(context)):
    label[k] = context[k]

###########################
# Read Wavfile and Make Predictions
###########################
x = wavfile_to_examples(selected_file)
with session.graph.as_default():
    set_session(session)

    x = x.reshape(len(x), 96, 64, 1)
    predictions = model.predict(x)

    for k in range(len(predictions)):
        prediction = predictions[k]
        m = np.argmax(prediction)
        print("Prediction: %s (%0.2f)" % (ubicoustics.to_human_labels[label[m]], prediction[m]))
