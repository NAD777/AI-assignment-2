from copy import deepcopy, copy
import mido
from mido import Message, MetaMessage
from math import sqrt
from random import randint, choice, random
from typing import List, Tuple
from tqdm import tqdm
import argparse

"""Constant that define the length of the one chord in accompaniment"""
BARLEN_DIVIDED_4 = 384

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

"""Shifts with them help we can generate: minor, major and others things for one note"""
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


def mido_to_note(mido_note):
    """Function that translates note from mido to the number of note on piano ignoring octave"""
    return mido_note % 12


class Chord:
    """
    Class that holds notes of chord, lad - major or minor and step - number of accord in enharmonic keys
    """

    def __init__(self, root_note, scale, step):
        """
        Constructor for Chord class
        :param root_note: root note of accords
        :param scale: minor or major
        :param step: step of chord in table (from assignment)
        """
        self.root_note = root_note
        self.scale = scale  # major or minor
        self.step = step  # step of note
        self.notes = [0, 0, 0]

    def get_triad(self):
        """
        Sets the notes of chord equal to the triad of this chord scale and root note
        :return: this object
        """
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS[self.scale]['triad']]
        return self

    def get_first_inversion(self):
        """
        Sets the notes of chord equal to the first inversion of this chord scale and root note
        :return: this object
        """
        self.notes = [self.root_note + offset for offset in OFFSETS[self.scale]['first']]
        return self

    def get_second_inversion(self):
        """
        Sets the notes of chord equal to the second inversion of this chord scale and root note
        :return: this object
        """
        self.notes = [self.root_note + offset for offset in OFFSETS[self.scale]['second']]
        return self

    def get_dim(self):
        """
        Sets the notes of chord equal to the diminished chord with root note on this chord
        :return: this object
        """
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS["diminished"]]
        return self

    def get_sus2(self):
        """
        Sets the notes of chord equal to the suspended 2 chord with root note on this chord
        :return: this object
        """
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS["sus2"]]
        return self

    def get_sus4(self):
        """
        Sets the notes of chord equal to the suspended 4 chord with root note on this chord
        :return: this object
        """
        self.notes = [(self.root_note + offset) % 12 for offset in OFFSETS["sus4"]]
        return self

    def is_major(self):
        """
        Check for scale of this chord
        :return: true if this chord in major scale else false
        """
        return self.scale == 'major'

    def is_minor(self):
        """
        Check for scale of this chord
        :return: true if this chord in minor scale else false
        """
        return self.scale == 'minor'

    def is_first_inversion(self):
        """
        Check is this chord is first inverse
        :return: true if this chord is first inverse else false
        """
        return self.notes == [self.root_note + offset for offset in OFFSETS[self.scale]['first']]

    def is_second_inversion(self):
        """
        Check is this chord is second inverse
        :return: true if this chord is second inverse else false
        """
        return self.notes == [self.root_note + offset for offset in OFFSETS[self.scale]['second']]

    def is_dim(self):
        """
        Check is this chord is diminished
        :return: true if this chord is diminished else false
        """
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["diminished"]]

    def is_sus2(self):
        """
        Check is this chord is suspended 2
        :return: true if this chord is suspended 2 else false
        """
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["sus2"]]

    def is_sus4(self):
        """
        Check is this chord is suspended 4
        :return: true if this chord is suspended 4 else false
        """
        return self.notes == [(self.root_note + offset) % 12 for offset in OFFSETS["sus4"]]

    def __str__(self):
        """
        Operator overload for str operation
        :return: return this chord in human-readable format (used mainly for debugging)
        """
        arr = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return arr[self.root_note % 12] + ('m' if self.scale == 'minor' else '') + (
            '˙' if self.is_dim() else "")

    def __contains__(self, note: int):
        """
        Operator overload for 'in' operation
        :param note: note in integer format
        :return: true if this chord contains note else false
        """
        return note in self.notes

    def __eq__(self, other):
        """
        Operator overload for equal sign
        :param other: other chord
        :return: true if other chord equal to this
        """
        return self.notes == other.notes


class MainKey:
    """
    The class that defines key of the given song.
    For defining the key of the song I am using statistic approach from https://rnhart.net/articles/key-finding/
    """
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    """Values for major profile of key defining algorithm"""
    major_profile = [("C", 17.7661), ("C#", 0.145624), ("D", 14.9265), ("D#", 0.160186), ("E", 19.8049), ("F", 11.3587),
                     ("F#", 0.291248),
                     ("G", 22.062), ("G#", 0.145624), ("A", 8.15494), ("A#", 0.232998), ("B", 4.95122)]

    """Values for minor profile of key defining algorithm"""
    minor_profile = [("A", 18.2648), ("A#", 0.737619), ("B", 14.0499), ("C", 16.8599), ("C#", 0.702494), ("D", 14.4362),
                     ("D#", 0.702494),
                     ("E", 18.6161), ("F", 4.56621), ("F#", 1.93186), ("G", 7.37619), ("G#", 1.75623)]

    def __init__(self, mido_file: mido.MidiFile):
        """
        Constructor for class MainKey, it finds duration of each note in song
        :param mido_file: opened file name in midi format
        """
        notes = []
        for i, track in enumerate(mido_file.tracks):
            for msg in track:
                notes.append(msg)
        self.duration = [0 for _ in range(12)]
        for el in notes:
            if el.type == "note_off":
                self.duration[mido_to_note(el.note)] += el.time
        self.duration = list(zip(self.note_names, self.duration))

    @staticmethod
    def __circle_permutation(array):
        """Supporting function that make circular permutation of array"""
        return array[1:] + [array[0]]

    @staticmethod
    def __calculate_correlation(x, y):
        """
        Calculates correlation formula for given x and y
        :param x: one coefficient from major or minor profile
        :param y: duration of the note
        :return: value of correlation
        """
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        sum_numerator = 0
        for i in range(len(x)):
            sum_numerator += (x[i] - mean_x) * (y[i] - mean_y)

        sum_denumirator_x = 0
        for i in range(len(x)):
            sum_denumirator_x += (x[i] - mean_x) ** 2

        sum_denum_y = 0
        for i in range(len(y)):
            sum_denum_y += (y[i] - mean_y) ** 2

        return sum_numerator / sqrt(sum_denumirator_x * sum_denum_y)

    def get_key(self):
        """
        Function that defines key.
        The key-finding algorithm calculates a correlation coefficient for each possible major and minor key by pairing
        the pitch class values to the profile values for the key in question
        :return: key of the song
        """
        max_r = -1e9
        is_major = False
        key_note = 0
        major_x = self.major_profile
        dur = copy(self.duration)
        for i in range(12):
            r = self.__calculate_correlation(list(map(lambda x: x[1], major_x)),
                                             list(map(lambda x: x[1], dur)))
            if max_r < r:
                max_r = r
                is_major = True
                key_note = i
            dur = self.__circle_permutation(dur)

        minor_x = self.minor_profile
        dur = copy(self.duration)
        for i in range(12):
            r = self.__calculate_correlation(list(map(lambda x: x[1], minor_x)),
                                             list(map(lambda x: x[1], dur)))
            if max_r < r:
                max_r = r
                is_major = False
                key_note = i
            dur = self.__circle_permutation(dur)

        return key_note, "major" if is_major else "minor"


class GoodChords:
    """Class that holds well-sounding chords for the given key"""

    """major table offsets from assignment"""
    major = [0, 2, 4, 5, 7, 9, 11]

    """minor table offsets from assignment"""
    minor = [0, 2, 3, 5, 7, 8, 10]

    def __init__(self, key_note):
        """
        Constructor for class GoodChords
        :param key_note: tuple of keynote in integer representation and scale of keynote
        """
        self.key_note, self.scale = key_note
        self.good_chords = self.get()

    def get(self):
        """
        Function that generates and returns well-sounding chords for keynote
        :return: list of Chord classes
        """
        offsets = self.major if self.scale == "major" else self.minor
        good_chords = []
        if self.scale == "major":
            good_chords.append(Chord((self.key_note + offsets[0]) % 12, "major", 1).get_triad())
            good_chords.append(Chord((self.key_note + offsets[1]) % 12, "minor", 2).get_triad())
            good_chords.append(Chord((self.key_note + offsets[2]) % 12, "minor", 3).get_triad())
            good_chords.append(Chord((self.key_note + offsets[3]) % 12, "major", 4).get_triad())
            good_chords.append(Chord((self.key_note + offsets[4]) % 12, "major", 5).get_triad())
            good_chords.append(Chord((self.key_note + offsets[5]) % 12, "minor", 6).get_triad())
            good_chords.append(Chord((self.key_note + offsets[6]) % 12, "major", 7).get_dim())
        if self.scale == "minor":
            good_chords.append(Chord((self.key_note + offsets[0]) % 12, "minor", 1).get_triad())
            good_chords.append(Chord((self.key_note + offsets[1]) % 12, "major", 2).get_dim())
            good_chords.append(Chord((self.key_note + offsets[2]) % 12, "major", 3).get_triad())
            good_chords.append(Chord((self.key_note + offsets[3]) % 12, "minor", 4).get_triad())
            good_chords.append(Chord((self.key_note + offsets[4]) % 12, "minor", 5).get_triad())
            good_chords.append(Chord((self.key_note + offsets[5]) % 12, "major", 6).get_triad())
            good_chords.append(Chord((self.key_note + offsets[6]) % 12, "major", 7).get_triad())
        return good_chords


class Song:
    """Class that holds information of the song and provide function to work with given melody"""
    def __init__(self, name_of_midi):
        """
        Constructor for Song class, that holds and does things related to initial melody that was given as input
        :param name_of_midi: source midi file name
        """
        self.mido_file = mido.MidiFile(name_of_midi, clip=True)
        self.key = MainKey(self.mido_file)
        self.begin = self.mido_file.tracks[1][2].time
        self.len_in_bars4 = 0
        self.divided = self.divide_track()

    def get_average_velocity(self):
        """
        Function that calculates the average velocity of the song
        :return: average velocity of song
        """
        list_of_velocities = list(map(lambda x: x.velocity,
                                      filter(lambda x: not x.is_meta and x.type == "note_on", self.mido_file)))
        return sum(list_of_velocities) // len(list_of_velocities)

    def get_tempo(self) -> int:
        """
        Function that return tempo of the song
        :return: tempo: int
        """
        element = list(map(lambda x: x,
                           filter(lambda x: x.is_meta and x.type == "set_tempo", self.mido_file)))[0]
        return element.tempo

    def get_average_octave(self) -> int:
        """
        Function that calculates the average octave of the song
        :return: average octave number
        """
        messages = []
        for track in self.mido_file.tracks:
            for msg in track:
                messages.append(msg)
        octaves = list(map(lambda x: x.note // 12, filter(lambda x: x.type == "note_on", messages)))
        return sum(octaves) // len(octaves)

    def get_key(self):
        """
        Getter for song key
        :return: key
        """
        return self.key.get_key()

    def divide_track(self):
        """
        Function that makes array of notes in every bar divided by 4 interval
        :return: array of notes in integer format in every bar divided by 4 interval
        """
        notes_on = list(map(lambda x: (mido_to_note(x.note), x.time),
                            filter(lambda x: x.type == 'note_on', self.mido_file.tracks[1][2:])))
        notes_off = list(map(lambda x: (mido_to_note(x.note), x.time),
                             filter(lambda x: x.type == 'note_off', self.mido_file.tracks[1][2:])))
        zipped = list(zip(notes_on, notes_off))
        sum_bars = 0
        for (note, btime), (note, etime) in zipped:
            sum_bars += btime + etime

        result = [(BARLEN_DIVIDED_4 * i, BARLEN_DIVIDED_4 * (i + 1), []) for i in
                  range(sum_bars // BARLEN_DIVIDED_4 + 1)]
        self.len_in_bars4 = sum_bars // BARLEN_DIVIDED_4
        cur_sum = 0
        for (note, btime), (note, etime) in zipped:
            cur_sum += btime
            save_cur_sum = cur_sum
            while cur_sum < save_cur_sum + etime:
                for ind, (begin, end, _) in enumerate(result):
                    if begin <= cur_sum < end:
                        result[ind][2].append(note)
                cur_sum += BARLEN_DIVIDED_4
            cur_sum = save_cur_sum + etime
        return list(map(lambda x: x[2], result))

    def save_with_accompaniment(self, chords: List[Tuple[int, int, int]], out_file_name: str):
        """
        Function that saves the song with given accompaniment cords
        :param chords: chords to add to accompaniment
        :param out_file_name: file name of output file
        """
        average_velocity = self.get_average_velocity()
        average_octave = self.get_average_octave()
        accompaniment = mido.MidiTrack()
        accompaniment.append(MetaMessage('track_name', name='Accompaniment', time=0))
        accompaniment.append(Message("program_change", channel=0, program=1, time=0))

        begin_index = 0
        while chords[begin_index][0] is None:
            begin_index += 1
        accompaniment.append(
            Message("note_on", channel=0, note=average_octave * 12 + chords[begin_index][0], velocity=average_velocity,
                    time=self.begin))
        accompaniment.append(
            Message("note_on", channel=0, note=average_octave * 12 + chords[begin_index][1], velocity=average_velocity,
                    time=0))
        accompaniment.append(
            Message("note_on", channel=0, note=average_octave * 12 + chords[begin_index][2], velocity=average_velocity,
                    time=0))

        accompaniment.append(
            Message("note_off", channel=0, note=average_octave * 12 + chords[begin_index][0], velocity=0,
                    time=BARLEN_DIVIDED_4 - self.begin % BARLEN_DIVIDED_4))
        accompaniment.append(
            Message("note_off", channel=0, note=average_octave * 12 + chords[begin_index][1], velocity=0, time=0))
        accompaniment.append(
            Message("note_off", channel=0, note=average_octave * 12 + chords[begin_index][2], velocity=0, time=0))

        flag = False
        shift = 0
        for first, second, third in chords[begin_index + 1:]:
            if first is None and second is None and third is None:
                shift += 1
                flag = True
                continue
            if flag:
                accompaniment.append(
                    Message("note_on", channel=0, note=average_octave * 12 + first, velocity=average_velocity,
                            time=shift * BARLEN_DIVIDED_4))
                shift = 0
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

        note_num, tonic = self.key.get_key()
        self.mido_file.save(f"{out_file_name}_{note_names[note_num]}_{tonic}.mid")


class Gene:
    """Class that defines the gene for genetic algorithm, in my case gene is one chord of the accompaniment"""

    def __init__(self, chord):
        """
        Gene constractor
        :param chord: chord of the gene
        """
        self.chord: Chord = chord

    def mutate(self):
        """
        Function that mutate current gene, we mutate only with 10% chance
        if chord is diminished -> no mutation at all
        Probability  mutation to
            1/4      the first inverse
            1/4      the second inverse
            1/4      the suspended 2
            1/4      the suspended 4
        """
        if random() < 0.1:
            return
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
        """
        Operator overload for 'in' operator
        :param item: specific not in integer value
        :return: true if Gene (Chord) contains note (item) else false
        """
        return item in self.chord

    def __eq__(self, other):
        """
        Operator overload for equals sign
        :param other: other gene (chord)
        :return: if this chord equals other return true else false
        """
        return self.chord == other.chord


class Chromosome:
    """
    Class that defines Chromosome of genetic algorithm, in my case chromosome is whole accompaniment for the song
    """
    genes: List[Gene] = []

    def __init__(self, genes: List[Gene]):
        """
        Constructor for class Chromosome
        :param genes: sequence of chords
        """
        self.genes = genes

    def fitness(self, divided_song, tonic_chords) -> int:
        """
        Fitness function that can evaluate this chromosome (accompaniment) based on different rules
        :param divided_song: array of notes in every bar divided by 4 interval
        :param tonic_chords: well-sounding chords for this song key
        :return: integer number that shows how good this accompaniment based, this evaluation based on different rules,
        that written in report
        """
        counter = 0

        # matching original quarter notes and generated
        for quarter, gen in zip(divided_song, self.genes):
            for note in quarter:
                if note in gen:
                    counter += 50

        # check for two sequential chords
        for i in range(len(self.genes) - 1):
            if self.genes[i] == self.genes[i + 1]:
                counter -= 20

        # if chords are not diminished and not sus2 and not sus4
        for gene in self.genes:
            if gene.chord.is_sus2() or gene.chord.is_sus4():
                counter -= 30

        # check for that first chords of accompaniment is key chord
        if self.genes[0].chord == tonic_chords[0]:
            counter += 50

        return counter

    def mutate(self):
        """
        Function that mutates all genes (chords) in this chromosome
        :return: this object
        """
        for i in range(len(self.genes)):
            self.genes[i].mutate()
        return self


class Generator:
    """Generator class that implements genetic algorithm"""

    def __init__(self, file_name: str, population_size: int, number_of_generations: int, out_file_name: str):
        """
        Constructor for Generator class
        :param file_name: input file name
        :param population_size: size of every population of algorithm
        :param number_of_generations: the number of people we must produce
        :param out_file_name: the name of output file
        """
        self.song: Song = Song(file_name)
        self.tonic_chords = GoodChords(self.song.get_key()).get()
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.out_file_name = out_file_name

    def create_initial_population(self):
        """
        Function that returns the initial population of genetic algorithm
        :return: array of Chromosomes
        """
        initial_population = [Chromosome([Gene(choice(self.tonic_chords)) for _ in range(self.song.len_in_bars4)])
                              for _ in range(self.population_size)]

        return initial_population

    @staticmethod
    def _crossover(first_chromosome: Chromosome, second_chromosome: Chromosome) -> Chromosome:
        """
        Function that merges two chromosomes randomly: it takes suffix of random length
        from first chromosome and complements the suffix of the second chromosome
        :param first_chromosome: chromosome
        :param second_chromosome: chromosome
        :return:
        """
        result = first_chromosome.genes[:randint(0, len(first_chromosome.genes) - 1)]
        result = result + second_chromosome.genes[len(result):]
        return Chromosome(result)

    def crossover_two_child(self, first_parent: Chromosome, second_parent: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Function that produces two children using crossover algorithm
        first child = crossover(parent 1, parent 2)
        second child = crossover(parent 2, parent 1)
        :param first_parent: Chromosome
        :param second_parent: Chromosome
        :return: two children
        """
        first_child = self._crossover(first_parent, second_parent)
        second_child = self._crossover(second_parent, first_parent)
        return first_child, second_child

    def get_population_fitness(self, population: List[Chromosome]) -> List[int]:
        """
        Function that calculates fitness function for each chromosome in population
        :param population: list of Chromosomes
        :return: list of ints every cell of it the value of fitness for corresponding chromosome[i]
        """
        return [chromosome.fitness(self.song.divided, self.tonic_chords) for chromosome in population]

    def next_population(self, prev_population: List[Chromosome]) -> List[Chromosome]:
        """
        Function that produces next population of genetic algorithm based on previous population
        makes selections, crossovers and mutations
        :param prev_population:
        :return: next population of genetic algorithm
        """
        new_population = prev_population
        zipped = list(sorted(zip(self.get_population_fitness(prev_population), prev_population), key=lambda x: -x[0]))

        best_parent1, best_parent2 = zipped[0][1], zipped[1][1]
        for _ in range(self.population_size):
            child1, child2 = self.crossover_two_child(best_parent1,
                                                      best_parent2)  # can try with random from prev_population
            new_population.append(deepcopy(child1).mutate())
            new_population.append(deepcopy(child2).mutate())

        zipped_huge_population = list(
            sorted(zip(self.get_population_fitness(new_population), new_population), key=lambda x: -x[0]))

        new_result_population = list(map(lambda x: x[1], zipped_huge_population))[:self.population_size]
        return new_result_population

    def generate(self):
        """
        Function that makes self.number_of_generations populations of genetic algorithm.
        And save song with accompaniment.
        """
        population = self.create_initial_population()
        for _ in tqdm(range(self.number_of_generations)):
            population = self.next_population(population)

        best_chromosome = population[0]
        best_chromosome_chords = list(map(lambda x: x.chord.notes, best_chromosome.genes))

        for i in range(len(best_chromosome_chords)):
            if len(self.song.divided[i]) == 0:
                best_chromosome_chords[i] = [None, None, None]
        self.song.save_with_accompaniment(best_chromosome_chords, self.out_file_name)


if __name__ == '__main__':
    """Argument command line parser for working with program through console"""
    parser = argparse.ArgumentParser(description='Accompaniment adder')
    parser.add_argument('file', type=str, help="Name of source file")
    parser.add_argument(
        '--population', '-n',
        type=int,
        default=600,
        help='Provide the size of initial and successive populations (default: 600)'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=100,
        help='Provide the amount of iterations (default: 100)'
    )

    parser.add_argument(
        '--out', '-o',
        type=str,
        default=None,
        help='Name of output file'
    )
    args = parser.parse_args()
    output_file_name = args.out if args.out is not None else f"out_{str(args.file).split('.')[0]}"
    Generator(args.file, args.population, args.iterations, output_file_name).generate()
