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

    def get_features(self, feature_type="window"):
        if feature_type == "window":
            features = self.window_features()
        else:
            raise ValueError("Unrecognised feature type {}".format(feature_type))

        return features

    @staticmethod
    def get_classifier(class_type="nn", classifier_params={}):
        if class_type == "nn":
            from sklearn import neighbors

            classifier_params.setdefault("weights", "distance")
            classifier_params.setdefault("n_neighbors", 10)
            clf = neighbors.KNeighborsClassifier(**classifier_params)

        elif class_type == "pipeline":
            from sklearn import preprocessing
            from sklearn.pipeline import make_pipeline
            from sklearn import svm

            clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

        else:
            raise ValueError("Unrecognised classifier type {}".format(class_type))

        return clf

    @staticmethod
    def get_cross_val_set(strategy="shuffle", cross_val_params={}):
        if strategy == "shuffle":
            from sklearn.model_selection import ShuffleSplit

            cross_val_params.setdefault("n_splits", 10)
            cross_val_params.setdefault("test_size", 0.2)
            cross_val_params.setdefault("random_state", 0)
            shuffle = ShuffleSplit(**cross_val_params)
        else:
            raise ValueError("Unrecognised cross validation {}".format(strategy))
        return shuffle

    def decode(self, class_type="nn"):
        """Decode by fitting with default parameters."""
        # TODO change the test set
        features = self.get_features()
        clf = self.get_classifier()
        to_predict = self.get_tags()
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

    def cross_val_decoding(
        self,
        class_type="nn",
        feature_type="window",
        cv_strategy="shuffle",
        cross_val_params={"n_splits": 10, "test_size": 0.2, "random_state": 0},
        scoring=["accuracy", "balanced_accuracy"],
        verbose=False,
    ):
        """
        Perform decoding with cross-validation.

        """
        from sklearn.model_selection import cross_validate

        cv = self.get_cross_val_set(strategy="shuffle")
        features = self.get_features(feature_type=feature_type)
        clf = self.get_classifier(class_type=class_type)
        tags = self.get_tags()
        print(
            "Running cross val on\n{}\nwith cv\n{}\nusing features: {}".format(
                clf, cv, feature_type
            )
        )
        return cross_validate(
            clf, features, tags, return_train_score=True, scoring=scoring, cv=cv
        )

    @staticmethod
    def confidence_interval_estimate(cross_val_result, key):
        test_key = "test_" + key
        train_key = "train_" + key
        test_scores = cross_val_result[test_key]
        train_scores = cross_val_result[train_key]

        test_str = "Test {}: {:.2f} (+/- {:.2f})".format(
            key, test_scores.mean(), test_scores.std() * 2
        )
        train_str = "Train {}: {:.2f} (+/- {:.2f})".format(
            key, train_scores.mean(), train_scores.std() * 2
        )
        return test_str, train_str


def random_decoding():
    from bvmpc.bv_mne import random_white_noise

    # Just random white noise signal
    random_epochs = random_white_noise(100, 4, 500)

    # Random one or zero tags for now
    tags = np.random.randint(low=0, high=2, size=100)
    target_names = ["Random OFF", "Random ON"]

    decoder = LFPDecoder(mne_epochs=random_epochs, tags=tags)
    out = decoder.decode()
    print("Actual :", tags[-10:])
    print("Predict:", out[1])
    print(decoder.decoding_accuracy(tags[-10:], out[1], target_names))

    print("----------Cross Validation-------------")
    from pprint import pprint

    cv_result = decoder.cross_val_decoding(
        cross_val_params={"n_splits": 1000, "test_size": 0.2, "random_state": None},
        verbose=True,
    )
    pprint(cv_result)
    pprint(decoder.confidence_interval_estimate(cv_result, "accuracy"))


if __name__ == "__main__":
    random_decoding()
