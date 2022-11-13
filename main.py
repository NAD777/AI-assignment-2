from copy import copy

import music21 as music

import mido
from mido import Message, MidiTrack, MetaMessage
from math import sqrt
from random import randint, choice
from typing import List, Tuple

# INP = "barbiegirl_mono.mid"
INP = "input1.mid"
# OUT = "output.mid"

res = music.converter.parse(INP)

# print("Tonic:", res.analyze('key'))
# print("Tonic:", res.analyze('key').tonic.midi)

BARLEN_DIVIDED_4 = 384


def mido_to_note(mido_note):
    return mido_note % 12


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


def circle_permutation(array):
    return array[1:] + [array[0]]


class Chord:
    def __init__(self, root_note, lad, step):
        self.root_note = root_note
        self.lad = lad  # major or minor
        self.step = step  # step of note
        self.notes = [0, 0, 0]

    def get_triad(self):
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS[self.lad]['triad']]
        return self

    def get_first_inversion(self):
        self.notes = [self.root_note + offset for offset in OFFSETS[self.lad]['first']]
        return self

    def get_second_inversion(self):
        self.notes = [self.root_note + offset for offset in OFFSETS[self.lad]['second']]
        return self

    def get_dim(self):
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS["diminished"]]
        return self

    def get_sus2(self):
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS["sus2"]]
        return self

    def get_sus4(self):
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS["sus4"]]
        return self

    def is_major(self):
        return self.lad == 'major'

    def is_minor(self):
        return self.lad == 'minor'

    def is_first_inversion(self):
        return self.notes == [self.root_note + offset for offset in OFFSETS[self.lad]['first']]

    def is_second_inversion(self):
        return self.notes == [self.root_note + offset for offset in OFFSETS[self.lad]['second']]

    def is_dim(self):
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["diminished"]]

    def is_sum2(self):
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["sus2"]]

    def is_sum4(self):
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["sus4"]]

    def __str__(self):
        arr = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return arr[self.root_note % 12] + ('m' if self.lad == 'minor' else '') + (
            '˙' if self.is_dim() else "")


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
        # print(*notes, sep="\n")
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

    def get_key(self):
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
        # print(self.duration)
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
        # print(note_names[key_note])
        return key_note, "major" if is_major else "minor"


class GoodChords:
    major = [0, 2, 4, 5, 7, 9, 11]
    minor = [0, 2, 3, 5, 7, 8, 10]
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, key_note):
        self.key_note, self.lad = key_note
        self.good_chords = self.get()

    def get(self):
        offsets = self.major if self.lad == "major" else self.minor
        good_chords = []
        if self.lad == "major":
            good_chords.append(Chord((self.key_note + offsets[0]) % 12, "major", 1).get_triad())
            good_chords.append(Chord((self.key_note + offsets[1]) % 12, "minor", 2).get_triad())
            good_chords.append(Chord((self.key_note + offsets[2]) % 12, "minor", 3).get_triad())
            good_chords.append(Chord((self.key_note + offsets[3]) % 12, "major", 4).get_triad())
            good_chords.append(Chord((self.key_note + offsets[4]) % 12, "major", 5).get_triad())
            good_chords.append(Chord((self.key_note + offsets[5]) % 12, "minor", 6).get_triad())
            good_chords.append(Chord((self.key_note + offsets[6]) % 12, "major", 7).get_dim())
        if self.lad == "minor":
            good_chords.append(Chord((self.key_note + offsets[0]) % 12, "minor", 1).get_triad())
            good_chords.append(Chord((self.key_note + offsets[1]) % 12, "major", 2).get_dim())
            good_chords.append(Chord((self.key_note + offsets[2]) % 12, "major", 3).get_triad())
            good_chords.append(Chord((self.key_note + offsets[3]) % 12, "minor", 4).get_triad())
            good_chords.append(Chord((self.key_note + offsets[4]) % 12, "minor", 5).get_triad())
            good_chords.append(Chord((self.key_note + offsets[5]) % 12, "major", 6).get_triad())
            good_chords.append(Chord((self.key_note + offsets[6]) % 12, "major", 7).get_triad())
        return good_chords

    def get_tuples(self):
        # for test, need changes
        return [el.get_triad() for el in self.good_chords]


class Song:
    def __init__(self, name_of_midi):
        self.mido_file = mido.MidiFile(name_of_midi, clip=True)
        # for el in self.mido_file:
        #     print(el)
        self.key = MainKey(self.mido_file)
        self.begin = self.mido_file.tracks[1][2].time
        self.len_in_bars4 = 0
        self.devided = self.divide_track()

    def get_average_velocity(self):
        list_of_velocities = list(map(lambda x: x.velocity,
                                      filter(lambda x: not x.is_meta and x.type == "note_on", self.mido_file)))
        return sum(list_of_velocities) // len(list_of_velocities)

    def get_tempo(self):
        element = list(map(lambda x: x,
                           filter(lambda x: x.is_meta and x.type == "set_tempo", self.mido_file)))[0]
        return element.tempo

    # in assignment was given that our track has 4/4 time signature
    def get_chord_duration(self):
        return self.mido_file.ticks_per_beat * 2

    def get_average_octave(self) -> int:
        amount = 0
        sum_octaves = 0
        for i, track in enumerate(self.mido_file.tracks):
            for msg in track:
                if msg.type == "note_on":
                    amount += 1
                    sum_octaves += msg.note // 12
        return sum_octaves // amount

    def get_key(self):
        return self.key.get_key()

    def divide_track(self):
        notes_on = list(map(lambda x: (mido_to_note(x.note), x.time),
                            filter(lambda x: x.type == 'note_on', self.mido_file.tracks[1][2:])))
        notes_off = list(map(lambda x: (mido_to_note(x.note), x.time),
                             filter(lambda x: x.type == 'note_off', self.mido_file.tracks[1][2:])))
        zipped = list(zip(notes_on, notes_off))
        # print(*zipped[:6], sep="\n")
        # zipped = zipped[:6]
        sum_bars = 0
        for (note, btime), (note, etime) in zipped:
            sum_bars += btime + etime

        result = [(BARLEN_DIVIDED_4 * i, BARLEN_DIVIDED_4 * (i + 1), []) for i in
                  range(sum_bars // BARLEN_DIVIDED_4 + 1)]
        self.len_in_bars4 = sum_bars // BARLEN_DIVIDED_4 + 1
        cur_sum = 0
        index = 0
        for (note, btime), (note, etime) in zipped:
            cur_sum += btime
            for ind, (begin, end, _) in enumerate(result):
                if begin <= cur_sum < end:
                    result[ind][2].append(note)
                    if cur_sum + etime > end:
                        result[(cur_sum + etime) // BARLEN_DIVIDED_4][2].append(note)
            cur_sum += etime
        # print(result)
        return list(map(lambda x: x[2], result))

    def save_with_accompaniment(self, chords):
        average_velocity = self.get_average_velocity()
        average_octave = self.get_average_octave()
        accompaniment = mido.MidiTrack()
        accompaniment.append(MetaMessage('track_name', name='Accompaniment', time=0))
        accompaniment.append(Message("program_change", channel=0, program=1, time=0))

        accompaniment.append(
            Message("note_on", channel=0, note=average_octave * 12 + chords[0][0], velocity=average_velocity,
                    time=self.begin))
        accompaniment.append(
            Message("note_on", channel=0, note=average_octave * 12 + chords[0][1], velocity=average_velocity, time=0))
        accompaniment.append(
            Message("note_on", channel=0, note=average_octave * 12 + chords[0][2], velocity=average_velocity, time=0))

        accompaniment.append(
            Message("note_off", channel=0, note=average_octave * 12 + chords[0][0], velocity=0, time=BARLEN_DIVIDED_4))
        accompaniment.append(
            Message("note_off", channel=0, note=average_octave * 12 + chords[0][1], velocity=0, time=0))
        accompaniment.append(
            Message("note_off", channel=0, note=average_octave * 12 + chords[0][2], velocity=0, time=0))

        for first, second, third in chords[1:]:
            accompaniment.append(
                Message("note_on", channel=0, note=average_octave * 12 + first, velocity=average_velocity, time=0))
            accompaniment.append(
                Message("note_on", channel=0, note=average_octave * 12 + second, velocity=average_velocity, time=0))
            accompaniment.append(
                Message("note_on", channel=0, note=average_octave * 12 + third, velocity=average_velocity, time=0))

            accompaniment.append(
                Message("note_off", channel=0, note=average_octave * 12 + first, velocity=0, time=BARLEN_DIVIDED_4))
            accompaniment.append(
                Message("note_off", channel=0, note=average_octave * 12 + second, velocity=0, time=0))
            accompaniment.append(
                Message("note_off", channel=0, note=average_octave * 12 + third, velocity=0, time=0))
        accompaniment.append(MetaMessage('end_of_track', time=0))
        self.mido_file.tracks.append(accompaniment)
        self.mido_file.save("1.midi")


class Gene:  # аккорд
    def __init__(self, chord):
        self.chord: Chord = chord

    def mutate(self):
        mutate_name = ["inv1", "inv2", "sus2", "sus4"]
        if self.chord.is_dim():
            return

        type_mutation = choice(mutate_name)

        if type_mutation == "sus2":
            if (self.chord.is_major() and self.chord.step not in [3, 7]) \
                    or (self.chord.is_minor() and self.chord.step not in [2, 5]):
                self.chord = self.chord.get_sus2()
        elif type_mutation == "sus4":
            if (self.chord.is_major() and self.chord.step not in [4, 7]) \
                    or (self.chord.is_minor() and self.chord.step not in [2, 6]):
                self.chord = self.chord.get_sus4()
        elif type_mutation == "inv1":
            self.chord = self.chord.get_first_inversion()
        elif type_mutation == "inv2":
            self.chord = self.chord.get_second_inversion()


class Chromosome:
    genes = []

    def __init__(self, genes):
        self.genes = genes

    def fitness(self) -> int:
        pass


class Generator:
    # populaton
    # кол-во поколений
    def __init__(self, file_name, population_size):
        self.song = Song(file_name)
        self.tonic_accords = GoodChords(self.song.get_key()).get()
        self.population_size = population_size

    def create_initial_population(self):
        initial_population = [Chromosome([Gene(choice(self.tonic_accords)) for _ in range(self.song.len_in_bars4)])
                              for _ in range(self.population_size)]

        return initial_population

    def crossover(self, first_chromosome, second_chromosome):
        result = first_chromosome.genes[:len(first_chromosome.genes) // 2]
        result = result + second_chromosome[len(result):]
        return Chromosome(result)


generator = Generator(INP, 10)
# a = generator.create_initial_population()


a = [1, 2, 3]
b = [3, 4, 5]


def cross(a, b):
    arr = a[:len(a) // 2]
    arr = arr + b[len(arr):]
    print(arr)


cross(a, b)

# song = Song(INP)
# print(song.get_key(), sep="\n")
# good_Chords = GoodChords(song.get_key())
# print("Good tuples:", good_Chords.get_tuples())
# print("\n")
# arr = [1, 2, 3]
# print(circle_permutation(arr))

# print(song.get_average_velocity())
# print(song.get_tempo())
# print(song.get_chord_duration())

# print("Octava", song.get_average_octave())
#
# print(song.divide_track())
# print(list(zip(range(len(note_names)), note_names)))
# song.save_with_accompaniment(good_Chords.get_tuples())
# print(file)
print()
# for el in file:
#     print(type(el), el)
# out = mido.MidiFile()
# track_out = mido.MidiTrack()
# out.tracks.append(track_out)
# track_out.append(Message('note_on', channel=0, note=68, velocity=50, time=0))
# track_out.append(Message('note_off', channel=0, note=68, velocity=50, time=500))
# out.save(OUT)

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
