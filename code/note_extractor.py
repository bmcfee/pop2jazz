#!/usr/bin/env python

import argparse
import librosa
import numpy as np
import ujson as json
import sys

HOP_LENGTH=512

def detect_onsets(y, sr, hop_length):
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    return onset_frames

def harmonify(y):
    return librosa.istft(librosa.decompose.hpss(librosa.stft(y))[0])

def note_frames(y, sr, hop_length, onsets, max_notes):
    
    # Get the harmonic waveform
    y = harmonify(y)
    
    # Compute log-power CQT
    CQT = librosa.logamplitude(np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length))**2, ref_power=np.max, top_db=40.0)
    
    # Aggregate.  Drop everything before the first event
    event_notes = librosa.feature.sync(CQT, onsets, aggregate=np.median)[:, 1:]
    
    # Compute the top-3 notes, offset by 12 (C1) for cqt fmin
    idx = np.argsort(event_notes, axis=0)[::-1]
    idx = idx[:max_notes]
    
    # Convert onset events to times
    times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    
    events = []
    
    for i, (s, t) in enumerate(zip(times[:-1], times[1:])):
        events.append([s, t, filter(lambda x: x[-1] > -40.0, zip(idx[:, i] + 12, event_notes[idx[:, i], i]))])
    
    return filter(lambda x: x[-1], events)


def process_arguments(args):
    parser = argparse.ArgumentParser(description="Extracts note events from a stemmed audio file")

    parser.add_argument('input_file', action='store', type=str, help='Path to input audio')
    parser.add_argument('output_file', action='store', type=str, help='Path to output json')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    parameters = process_arguments(sys.argv[1:])

    y, sr = librosa.load(parameters['input_file'])

    onsets = detect_onsets(y, sr, HOP_LENGTH)

    notes = note_frames(y, sr, HOP_LENGTH, onsets, 3)

    with open(parameters['output_file'], 'w') as f:
        json.dump(notes, f)
