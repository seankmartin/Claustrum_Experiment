"""This module handles decoding routines."""

import numpy as np


class LFPDecoder(object):
    """
    Decode a dependent var x from indep LFP features.

    In general, this should be performed as
    1. Compute features from LFP
    2. Select the dependent variable (e.g. trial type)
    3. Perform classification or regression.

    Attributes
    ----------
    mne_epochs : mne.Epochs
        Epochs of LFP data.
        There should be an epoch for each dependent var.
        This is used to get
        a 3D array of shape (n_epochs, n_channels, n_times)
        For calculations.
    tags : np.ndarray
        The tag of each epoch, or what is decoded.
    selected_data : str | list | slice | None
        See mne.Epochs.get_data
        This is the picks arguments
    sample_rate : int
        The sampling rate of the lfp data.

    """

    def __init__(self, mne_epochs=None, tags=None, selected_data=None, sample_rate=250):
        self.mne_epochs = mne_epochs
        self.tags = tags
        self.selected_data = selected_data
        self.sample_rate = sample_rate

    def get_tags(self):
        return self.tags

    def get_data(self):
        return self.mne_epochs.get_data(picks=self.selected_data)

    def time_to_samples(self, time):
        """time is expected to be in seconds."""
        return time * self.sample_rate

    def window_features(self, window_sample_len=10, step=8):
        """Compute features from LFP in windows."""
        from skimage.util import view_as_windows

        data = self.get_data()

        # For now I'm just going to use non overlapping windows
        # And take the average of the power in that window
        # But you could use overlapping windows or other things
        if (data.shape[2] - window_sample_len) % step != 0:
            print(
                "WARNING: {} is not divisible by {} in window_features".format(
                    data.shape[2] - window_sample_len, step
                )
            )
        n_features = ((data.shape[2] - window_sample_len) // np.array(step)) + 1
        features = np.zeros(shape=(data.shape[0], n_features), dtype=np.float64)

        # For now, I'm going to take the average over the channels
        squished_data = np.mean(data, axis=1)

        # Performed overlapping windowing
        windowed_data = view_as_windows(
            squished_data, [1, window_sample_len], step=[1, step]
        ).squeeze()

        # For now I'll simply sum the window, but many things could be applied
        np.mean(windowed_data, axis=-1, out=features)

        return features

    def decode(self, class_type="nn"):
        """
        1. Compute features

        2. Select dependent var

        3. Perform ML
        """
        # This demonstrates really simple leave 10 out cross val
        # Will be update to use proper cross val later
        features = self.window_features()
        to_predict = self.get_tags()

        # You can use whatever here I'll just switch on type for now
        if class_type == "nn":
            from sklearn import neighbors

            clf = neighbors.KNeighborsClassifier(5, weights="distance")
        else:
            raise ValueError("Unrecognised classifier type {}".format(class_type))
        clf.fit(features[:-10], to_predict[:-10])
        output = clf.predict(features[-10:])

        return clf, output

    @staticmethod
    def decoding_accuracy(true, predicted, target_names=None, as_dict=False):
        """
        A report on decoding accuracy from true and predicted.

        Target names indicates the name of the tags (usually 0, 1, 2...)
        """
        from sklearn.metrics import classification_report

        return classification_report(
            true, predicted, target_names=target_names, output_dict=as_dict
        )


if __name__ == "__main__":
    from bvmpc.bv_mne import random_white_noise

    # Just random white noise signal
    random_epochs = random_white_noise(100, 4, 500)

    # Random one or zero tags for now
    tags = np.random.randint(low=0, high=2, size=100)
    target_names = ["Random OFF", "Random ON"]

    decoder = LFPDecoder(mne_epochs=random_epochs, tags=tags)
    out = decoder.decode()
    print(tags[-10:])
    print(out[1])
    print(decoder.decoding_accuracy(tags[-10:], out[1], target_names))
