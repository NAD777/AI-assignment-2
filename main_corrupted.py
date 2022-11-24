from copy import copy

# import music21 as music

import mido
from mido import Message, MidiTrack, MetaMessage
from math import sqrt
from random import randint, choice, shuffle
from typing import List, Tuple, Type
from copy import deepcopy
from tqdm import tqdm

INP = "barbiegirl_mono.mid"
# INP = "input3.mid"
# OUT = "output.mid"

# res = music.converter.parse(INP)

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

    def is_sus2(self):
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["sus2"]]

    def is_sus4(self):
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["sus4"]]

    def __str__(self):
        arr = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return arr[self.root_note % 12] + ('m' if self.lad == 'minor' else '') + (
            '˙' if self.is_dim() else "")

    def __contains__(self, note: int):
        return note in self.notes

    def __eq__(self, other):
        return self.notes == other.notes


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
        self.duration = [0 for _ in range(12)]
        for el in notes:
            if el.type == "note_off":
                self.duration[mido_to_note(el.note)] += el.time
        self.duration = list(zip(self.note_names, self.duration))

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
        # d = dict()
        # for el in self.note_names:
        #     d[el] = -1e9
        #     d[el + 'm'] = -1e9
        major_x = self.major_profile
        dur = copy(self.duration)
        for i in range(12):
            r = self._calculate_correlation(list(map(lambda x: x[1], major_x)),
                                            list(map(lambda x: x[1], dur)))
            # d[dur[0][0]] = max(d[dur[0][0]], r)
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
            # d[dur[0][0] + 'm'] = max(d[dur[0][0] + 'm'], r)
            if max_r < r:
                max_r = r
                is_major = False
                key_note = i
            dur = circle_permutation(dur)

        # key_sym = list(sorted(d.items(), key=lambda x: -x[1]))[0][0]
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
        self.divided = self.divide_track()

    def get_average_velocity(self):
        list_of_velocities = list(map(lambda x: x.velocity,
                                      filter(lambda x: not x.is_meta and x.type == "note_on", self.mido_file)))
        return sum(list_of_velocities) // len(list_of_velocities)

    def get_tempo(self):
        element = list(map(lambda x: x,
                           filter(lambda x: x.is_meta and x.type == "set_tempo", self.mido_file)))[0]
        return element.tempo

    # # in assignment was given that our track has 4/4 time signature
    # def get_chord_duration(self):
    #     return self.mido_file.ticks_per_beat * 2

    def get_average_octave(self) -> int:
        amount = 0
        sum_octaves = 0
        for i, track in enumerate(self.mido_file.tracks):
            for msg in track:
                if msg.type == "note_on":
                    amount += 1
                    sum_octaves += msg.note // 12
        return max(0, sum_octaves // amount - 1)
        # return sum_octaves // amount

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
        self.len_in_bars4 = sum_bars // BARLEN_DIVIDED_4
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

    def save_with_accompaniment(self, chords, out_file_name: str):
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

        flag = False
        for first, second, third in chords[1:]:
            if first is None and second is None and third is None:
                flag = True
                continue
            if flag:
                accompaniment.append(
                    Message("note_on", channel=0, note=average_octave * 12 + first, velocity=average_velocity,
                            time=BARLEN_DIVIDED_4))
                accompaniment.append(
                    Message("note_on", channel=0, note=average_octave * 12 + second, velocity=average_velocity, time=0))
                accompaniment.append(
                    Message("note_on", channel=0, note=average_octave * 12 + third, velocity=average_velocity, time=0))
                flag = False
            else:
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
        self.mido_file.save(f"{out_file_name}.midi")


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

    def __contains__(self, item: int):
        return item in self.chord

    def __eq__(self, other):
        return self.chord == other.chord


class Chromosome:
    genes: List[Gene] = []

    def __init__(self, genes):
        self.genes = genes

    def fitness(self, divided_song, tonic_chords) -> int:
        counter = 0

        # matching original quarter notes and generated

        for quarter, gen in zip(divided_song, self.genes):
            flag_no_match = True
            for note in quarter:
                if note in gen:
                    flag_no_match = True
                    counter += 12
            counter -= (10 if flag_no_match else 0)

        # check for two sequential chords
        for i in range(len(self.genes) - 1):
            if self.genes[i] == self.genes[i + 1]:
                counter -= 10
            else:
                counter += 15

        # check for tonic
        for gene in self.genes:
            if gene.chord in tonic_chords:
                counter += 6

        # if chords are not diminished and not sus2 and not sus4
        for gene in self.genes:
            if not gene.chord.is_dim() and not gene.chord.is_sus2() and not gene.chord.is_sus4():
                counter += 20
            else:
                counter -= 10

        # if song is empty and prev chord == current chord
        for i in range(1, len(self.genes)):
            if len(divided_song[i]) == 0 and self.genes[i] == self.genes[i - 1]:
                counter += 10
            else:
                counter += 4

        for i in range(len(self.genes) - 2):
            first, second, third = self.genes[i].chord, self.genes[i + 1].chord, self.genes[i + 2].chord
            if first == tonic_chords[6] and second == tonic_chords[1] and third == tonic_chords[0]:
                counter += 25
            if first == tonic_chords[1] and second == tonic_chords[6] and third == tonic_chords[0]:
                counter += 25

        if self.genes[0].chord == tonic_chords[0]:
            counter += 50
        else:
            counter -= 10

        for i in range(len(self.genes) - 1):
            first = self.genes[i].chord.notes
            second = self.genes[i + 1].chord.notes
            uni = set(first).union(set(second))
            if len(uni) == 1:
                counter += 25
            if len(uni) == 2:
                counter += 10
            if len(uni) == 0:
                counter -= 10

        return counter

    def mutate(self):
        # kind = choice(["mutate", "mutate", "shuffle"])
        # if kind == "mutate":
        for i in range(len(self.genes)):
            self.genes[i].mutate()
        # else:
        #     shuffle(self.genes)
        return self

    def copy(self):
        return Chromosome(self.genes)


class Generator:
    # populaton
    # кол-во поколений
    def __init__(self, file_name: str, population_size: int, number_of_generations: int, out_file_name: str):
        self.song: Song = Song(file_name)
        self.tonic_chords = GoodChords(self.song.get_key()).get()
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.out_file_name = out_file_name

        # self.generate()

    def create_initial_population(self):
        initial_population = [Chromosome([Gene(choice(self.tonic_chords)) for _ in range(self.song.len_in_bars4)])
                              for _ in range(self.population_size)]

        return initial_population

    def _crossover(self, first_chromosome, second_chromosome) -> Chromosome:
        result = first_chromosome.genes[:randint(0, len(first_chromosome.genes) - 1)]
        result = result + second_chromosome.genes[len(result):]
        return Chromosome(result)

    def crossover_two_child(self, first_parent: Chromosome, second_parent: Chromosome) -> Tuple[Chromosome, Chromosome]:
        first_child = self._crossover(first_parent, second_parent)
        second_child = self._crossover(second_parent, first_parent)
        return first_child, second_child

    def get_population_fitness(self, population: List[Chromosome]) -> List[int]:
        return [chromosome.fitness(self.song.divided, self.tonic_chords) for chromosome in population]

    def next_population(self, prev_population: List[Chromosome]) -> List[Chromosome]:
        # new_population = deepcopy(prev_population)
        new_population = prev_population

        zipped = list(sorted(zip(self.get_population_fitness(prev_population), prev_population), key=lambda x: -x[0]))

        best_parent1, best_parent2 = zipped[0][1], zipped[1][1]
        for _ in range(self.population_size):
            child1, child2 = self.crossover_two_child(best_parent1,
                                                      best_parent2)  # can try with random from prev_population
            new_population.append(child1)
            new_population.append(child2)
            new_population.append(child1.copy().mutate())
            new_population.append(child2.copy().mutate())

        zipped_huge_population = list(
            sorted(zip(self.get_population_fitness(new_population), new_population), key=lambda x: -x[0]))
        new_result_population = list(map(lambda x: x[1], zipped_huge_population))[:self.population_size]

        return new_result_population

    def generate(self):
        population = self.create_initial_population()
        for _ in tqdm(range(self.number_of_generations)):
            population = self.next_population(population)

        best_chromosome = population[0]
        best_chromosome_chords = list(map(lambda x: x.chord.notes, best_chromosome.genes))
        for i in range(len(best_chromosome_chords)):
            if len(self.song.divided[i]) == 0:
                best_chromosome_chords[i] = [None, None, None]
        # print(best_chromosome_chords[0])
        # print(self.song.get_key())
        # print(self.tonic_chords[0])
        self.song.save_with_accompaniment(best_chromosome_chords, self.out_file_name)


# a = generator.create_initial_population()

generator = Generator(INP, 1000, 300, "1")  # размер популяции, кол-во поколений
generator.generate()
# s = Song(INP)
# ind, t = s.get_key()
# print(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][ind], t)

import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Accompaniment adder')
#     parser.add_argument('file', type=str, help="Name of source file")
#     parser.add_argument(
#         '--population', '-n',
#         type=int,
#         default=600,
#         help='Provide the size of initial population (default: 600)'
#     )
#     parser.add_argument(
#         '--iterations', '-i',
#         type=int,
#         default=100,
#         help='Provide the amount of iterations (default: 100)'
#     )
#
#     parser.add_argument(
#         '--out', '-o',
#         type=str,
#         default=None,
#         help='Name of output file'
#     )
#     args = parser.parse_args()
#     print(args.file)
#     print(args.population)
#     print(args.iterations)
#     print(args.out)
#     output_file_name = args.out if args.out is not None else f"out_{str(args.file).split('.')[0]}"
#     Generator(INP, args.population, args.iterations, output_file_name).generate()

# a = [1, 2, 3]
# b = [3, 4, 5]
# print(a[:3])


# def cross(a, b):
#     arr = a[:len(a) // 2]
#     arr = arr + b[len(arr):]
#     print(arr)


# cross(a, b)

# gene1 = Gene(Chord(0, "major", 1).get_triad())
# gene2 = Gene(Chord(0, "major", 1).get_triad())
# print(gene1.chord == gene2.chord)

# print(randint(1, 2))
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
# print()
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
