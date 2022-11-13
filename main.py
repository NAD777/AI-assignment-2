from copy import copy

import music21 as music

import mido
from mido import Message
from math import sqrt

INP = "barbiegirl_mono.mid"
# INP = "input3.mid"
# OUT = "output.mid"

res = music.converter.parse(INP)

print("Tonic:", res.analyze('key'))
print("Tonic:", res.analyze('key').tonic.midi)

file = mido.MidiFile(INP, clip=True)

BARLEN_devide_4 = 384

OFFSETS = {
    'minor': {
        'triad': [0, 3, 7],
        'first': [3, 7, 12],
        'second': [7, 12, 15]
    },
    'major': {
        'triad': [0, 4, 7],
        'first': [4, 7, 12],
        'second': [7, 12, 16]
    },

    'diminished': [0, 3, 6],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7]
}
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def get_average_velocity(mido_file: mido.MidiFile):
    list_of_velocities = list(map(lambda x: x.velocity,
                                  filter(lambda x: not x.is_meta and x.type == "note_on", mido_file)))
    return sum(list_of_velocities) / len(list_of_velocities)


def get_tempo(mido_file: mido.MidiFile):
    element = list(map(lambda x: x,
                       filter(lambda x: x.is_meta and x.type == "set_tempo", mido_file)))[0]
    return element.tempo


# in assignment was given that our track has 4/4 time signature
def get_chord_duration(mido_file: mido.MidiFile):
    return mido_file.ticks_per_beat * 2


def mido_to_note(mido_note: int):
    return mido_note % 12


def circle_permutation(array):
    return array[1:] + [array[0]]


class Chord:
    def __init__(self, root_note, name_offset, is_diminished=False):
        self.root_note = root_note
        self.name_offset = name_offset
        self.is_diminished = is_diminished

    def get_triad(self):
        return [(self.root_note + offset) % 12 for offset in OFFSETS[self.name_offset]['triad']]

    def get_first_inversion(self):
        return [(self.root_note + offset) % 12 for offset in OFFSETS[self.name_offset]['first']]

    def get_second_inversion(self):
        return [(self.root_note + offset) % 12 for offset in OFFSETS[self.name_offset]['second']]

    def get_dim(self):
        return [(self.root_note + offset) % 12 for offset in OFFSETS["diminished"]]

    def get_sus2(self):
        return [(self.root_note + offset) % 12 for offset in OFFSETS["sus2"]]

    def get_sus4(self):
        return [(self.root_note + offset) % 12 for offset in OFFSETS["sus4"]]

    def __str__(self):
        arr = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return arr[self.root_note % 12] + ('m' if self.name_offset == 'minor' else '') + (
            '˙' if self.is_diminished else "")


class MainKey:
    notes = []
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    # major_profile = [("C", 6.35), ("C#", 2.23), ("D", 3.48), ("D#", 2.33), ("E", 4.38), ("F", 4.09), ("F#", 2.52),
    #                  ("G", 5.19), ("G#", 2.39), ("A", 3.66), ("A#", 2.29), ("B", 2.88)]
    # minor_profile = [("A", 6.33), ("A#", 2.68), ("B", 3.52), ("C", 5.38), ("C#", 2.60), ("D", 3.53), ("D#", 2.54),
    #                  ("E", 4.75), ("F", 3.98), ("F#", 2.69), ("G", 3.34), ("G#", 3.17)]

    # minor_profile = [("C", 6.33), ("C#", 2.68), ("D", 3.52), ("D#", 5.38), ("E", 2.60), ("F", 3.53), ("F#", 2.54),
    #                  ("G", 4.75), ("G#", 3.98), ("A", 2.69), ("A#", 3.34), ("B", 3.17)]

    major_profile = [("C", 17.7661), ("C#", 0.145624), ("D", 14.9265), ("D#", 0.160186), ("E", 19.8049), ("F", 11.3587),
                     ("F#", 0.291248),
                     ("G", 22.062), ("G#", 0.145624), ("A", 8.15494), ("A#", 0.232998), ("B", 4.95122)]

    minor_profile = [("A", 18.2648), ("A#", 0.737619), ("B", 14.0499), ("C", 16.8599), ("C#", 0.702494), ("D", 14.4362),
                     ("D#", 0.702494),
                     ("E", 18.6161), ("F", 4.56621), ("F#", 1.93186), ("G", 7.37619), ("G#", 1.75623)]

    def __init__(self, mido_file: mido.MidiFile):
        notes = []
        for i, track in enumerate(mido_file.tracks):
            for msg in track:
                notes.append(msg)
        print(*notes, sep="\n")
        self.duration = [0 for _ in range(12)]
        for el in notes:
            if el.type == "note_off":
                self.duration[mido_to_note(el.note)] += el.time
        self.duration = list(zip(self.note_names, self.duration))
        # self.duration = list(zip(self.note_names, [432, 231, 0, 405, 12, 316, 4, 126, 612, 0, 191, 1]))

    def _calculate_correlation(self, x, y):
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        sum_numerator = 0
        for i in range(len(x)):
            sum_numerator += (x[i] - mean_x) * (y[i] - mean_y)

        sum_denum_x = 0
        for i in range(len(x)):
            sum_denum_x += (x[i] - mean_x) ** 2

        sum_denum_y = 0
        for i in range(len(y)):
            sum_denum_y += (y[i] - mean_y) ** 2

        return sum_numerator / sqrt(sum_denum_x * sum_denum_y)

    def get_keys(self):
        '''
        http://rnhart.net/articles/key-finding/
        :return: key
        '''
        max_r = -1e9
        is_major = False
        key_note = 0
        d = dict()
        for el in self.note_names:
            d[el] = -1e9
            d[el + 'm'] = -1e9
        major_x = self.major_profile
        print(self.duration)
        dur = copy(self.duration)
        for i in range(12):
            r = self._calculate_correlation(list(map(lambda x: x[1], major_x)),
                                            list(map(lambda x: x[1], dur)))
            d[dur[0][0]] = max(d[dur[0][0]], r)
            if max_r < r:
                max_r = r
                is_major = True
                key_note = i
            dur = circle_permutation(dur)

        minor_x = self.minor_profile
        dur = copy(self.duration)
        for i in range(12):
            r = self._calculate_correlation(list(map(lambda x: x[1], minor_x)),
                                            list(map(lambda x: x[1], dur)))
            d[dur[0][0] + 'm'] = max(d[dur[0][0] + 'm'], r)
            if max_r < r:
                max_r = r
                is_major = False
                key_note = i
            dur = circle_permutation(dur)

        key_sym = list(sorted(d.items(), key=lambda x: -x[1]))[0][0]
        print(note_names[key_note])
        return key_note, "major" if is_major else "minor"


class GoodChords:
    major = [0, 2, 4, 5, 7, 9, 11]
    minor = [0, 2, 3, 5, 7, 8, 10]
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, key_note):
        self.key_note, self.lad = key_note

    def get(self):
        offsets = self.major if self.lad == "major" else self.minor
        good_notes = []
        if self.lad == "major":
            good_notes.append(Chord((self.key_note + offsets[0]) % 12, "major"))
            good_notes.append(Chord((self.key_note + offsets[1]) % 12, "minor"))
            good_notes.append(Chord((self.key_note + offsets[2]) % 12, "minor"))
            good_notes.append(Chord((self.key_note + offsets[3]) % 12, "major"))
            good_notes.append(Chord((self.key_note + offsets[4]) % 12, "major"))
            good_notes.append(Chord((self.key_note + offsets[5]) % 12, "minor"))
            good_notes.append(Chord((self.key_note + offsets[6]) % 12, "major", is_diminished=True))
        if self.lad == "minor":
            good_notes.append(Chord((self.key_note + offsets[0]) % 12, "minor"))
            good_notes.append(Chord((self.key_note + offsets[1]) % 12, "major", is_diminished=True))
            good_notes.append(Chord((self.key_note + offsets[2]) % 12, "major"))
            good_notes.append(Chord((self.key_note + offsets[3]) % 12, "minor"))
            good_notes.append(Chord((self.key_note + offsets[4]) % 12, "minor"))
            good_notes.append(Chord((self.key_note + offsets[5]) % 12, "major"))
            good_notes.append(Chord((self.key_note + offsets[6]) % 12, "major"))
        for el in good_notes:
            print(el, end=" ")


main_key = MainKey(file)
print(main_key.get_keys(), sep="\n")
good_Chords = GoodChords((0, "minor"))
good_Chords.get()
print("\n")
# arr = [1, 2, 3]
# print(circle_permutation(arr))

print(get_average_velocity(file))
print(get_tempo(file))
print(get_chord_duration(file))

# print(file)
print()
# for el in file:
#     print(type(el), el)
out = mido.MidiFile()
track_out = mido.MidiTrack()
out.tracks.append(track_out)
track_out.append(Message('note_on', channel=0, note=68, velocity=50, time=0))
track_out.append(Message('note_off', channel=0, note=68, velocity=50, time=500))
# out.save(OUT)

arr = [0]
for el in [2, 1, 2, 2, 1, 2, 2]:
    arr.append(arr[-1] + el)
print(arr)
"""
    Message('note_on', channel=0, note=68, velocity=50, time=0),
    Message('note_on', channel=0, note=64, velocity=50, time=0),
    Message('note_on', channel=0, note=68, velocity=50, time=0),
    Message('note_on', channel=0, note=73, velocity=50, time=0),
    Message('note_on', channel=0, note=69, velocity=50, time=0),
    Message('note_on', channel=0, note=66, velocity=50, time=384),
    Message('note_on', channel=0, note=63, velocity=50, time=0),
    Message('note_on', channel=0, note=66, velocity=50, time=0),
    Message('note_on', channel=0, note=71, velocity=50, time=0),
    Message('note_on', channel=0, note=68, velocity=50, time=0),
    Message('note_on', channel=0, note=66, velocity=50, time=0),
    Message('note_on', channel=0, note=64, velocity=50, time=0),
    Message('note_on', channel=0, note=64, velocity=50, time=384),
    Message('note_on', channel=0, note=61, velocity=50, time=0),
    Message('note_on', channel=0, note=66, velocity=50, time=0),
    Message('note_on', channel=0, note=61, velocity=50, time=0),
    Message('note_on', channel=0, note=66, velocity=50, time=384),
    Message('note_on', channel=0, note=64, velocity=50, time=0),
    Message('note_on', channel=0, note=68, velocity=50, time=0),
    Message('note_on', channel=0, note=66, velocity=50, time=0),
    20

"""
