import pickle


class ObjectSerializer:
    """This class can write and read genomes, so that they can be used later in the program"""

    @classmethod
    def serialize(cls, object, file_name):
        """This class serializes the given object in the given file name with .pickle behind it."""
        with open(file_name + '.pickle', 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_name):
        """This method loads the object in the given pickle file."""
        with open(file_name + '.pickle', 'rb') as handle:
            return pickle.load(handle)
