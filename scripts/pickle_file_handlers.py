from pickle import dump, load, HIGHEST_PROTOCOL


class HandlePickle():
    def __init__(self) -> None:
        """
        Initialize the Class assigning its object a new data holding dicitonary
        Parameters
        """
        self.data = {}

    def add_data(self, name: str, data) -> None:
        """
        Adds data with the specified name to its data dictionary
        Parameters
        """
        self.data[name] = data

    def save_data(self, file_name: str) -> None:
        """
        Saves all the data it collected in its data dictionary on a pickle file
        Parameters
        """
        with open(file_name, 'wb') as handle:
            dump(self.data, handle, protocol=HIGHEST_PROTOCOL)

    def load_data(self, file_name: str) -> None:
        """
        Loads all the data saved in a pickle file to its data dictionary
        Parameters
        """
        with open(file_name, 'rb') as handle:
            self.data = load(handle)

    def get_data(self) -> dict:
        """
        A function which returns the objects data dicitonary value
        Parameters

        """
        return self.data
