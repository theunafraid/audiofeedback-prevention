# audiofeedback-prevention

Simple model to score single raw channel audio data.
Output will be scores for 3 classes : normal speech, echo and howl.

Input 300ms of 16khz of raw audio data.

Trained with small amount of data from TCD VoIP dataset and MLCommons.

TODO QRNN ver.
TODO wasm ver.

At the time it was tested tensorflow-lite did not support RNN operators unless adapter/operator conversion code was written so the model was converted to tensorflowjs instead of wasm.
