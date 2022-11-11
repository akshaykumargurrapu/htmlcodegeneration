from __future__ import print_function

__author__ = 'Taneem Jan, taneemishere.github.io'


import os
from .. import Vocabulary as v
from .. import Utils as u
from ..model import Config as c


class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words = []

        self.voc = v.Vocabulary()
        self.size = 0

    @staticmethod
    def load_paths_only(path):
        print("Parsing data...")
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                path_gui = "{}/{}".format(path, f)
                gui_paths.append(path_gui)
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    path_img = "{}/{}.png".format(path, file_name)
                    img_paths.append(path_img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    path_img = "{}/{}.npz".format(path, file_name)
                    img_paths.append(path_img)

        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths

    def load(self, path, generate_binary_sequences=False):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    img = u.Utils.get_preprocessed_img("{}/{}.png".format(path, file_name), 256)
                    self.append(file_name, gui, img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    img = u.np.load("{}/{}.npz".format(path, file_name))["features"]
                    self.append(file_name, gui, img)

        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
        self.next_words = self.sparsify_labels(self.next_words, self.voc)
        if generate_binary_sequences:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)
        else:
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)

        self.size = len(self.ids)
        try:
            assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
            assert self.voc.size == len(self.voc.vocabulary)
        except:
            print("cv")

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))

        try:
            self.input_shape = self.input_images[0].shape
        except:
            print("hhi")
        self.output_size = self.voc.size

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

    def convert_arrays(self):
        print("Convert arrays...")
        self.input_images = u.np.array(self.input_images)
        self.partial_sequences = u.np.array(self.partial_sequences)
        self.next_words = u.np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = u.np.array(pic, dtype=u.np.uint8)
            u.Utils.show(pic)

        token_sequence = [c.START_TOKEN]
        for line in gui:
            line = line.replace(",", " ,").replace("\n", " \n")
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)
        token_sequence.append(c.END_TOKEN)

        suffix = [c.PLACEHOLDER] * c.CONTEXT_LENGTH

        a = u.np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - c.CONTEXT_LENGTH):
            context = a[j:j + c.CONTEXT_LENGTH]
            label = a[j + c.CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)

    @staticmethod
    def indexify(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.vocabulary[token])
            temp.append(u.np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                try:
                    sparse_vectors_sequence.append(voc.binary_vocabulary[token])
                except:
                    print("zx")
            temp.append(u.np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            try:
                temp.append(voc.binary_vocabulary[label])
            except:
                print("as")

        return temp

    def save_metadata(self, path):
        u.np.save("{}/meta_dataset".format(path), u.np.array([self.input_shape, self.output_size, self.size]))
