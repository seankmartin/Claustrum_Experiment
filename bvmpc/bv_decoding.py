"""This module handles decoding routines."""

import numpy as np
import scipy.stats


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
    labels : np.ndarray
        The tag of each epoch, or what is decoded.
    selected_data : str | list | slice | None
        See mne.Epochs.get_data
        This is the picks arguments
    sample_rate : int
        The sampling rate of the lfp data.

    """

    def __init__(
        self, mne_epochs=None, labels=None, selected_data=None, sample_rate=250
    ):
        self.mne_epochs = mne_epochs
        self.labels = labels
        self.selected_data = selected_data
        self.sample_rate = sample_rate

    def get_labels(self):
        return self.labels

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

    def get_features(self, feature_type="window", feature_params={}):
        if feature_type == "window":
            features = self.window_features(**feature_params)
        else:
            raise ValueError("Unrecognised feature type {}".format(feature_type))

        return features

    @staticmethod
    def get_classifier(class_type="nn", classifier_params={}, return_param_dist=False):
        if class_type == "nn":
            from sklearn import neighbors

            classifier_params.setdefault("weights", "distance")
            classifier_params.setdefault("n_neighbors", 10)
            clf = neighbors.KNeighborsClassifier(**classifier_params)

            param_dist = {
                "n_neighbors": scipy.stats.randint(3, 12),
                "weights": ("uniform", "distance"),
            }

        elif class_type == "pipeline":
            from sklearn import preprocessing
            from sklearn.pipeline import make_pipeline
            from sklearn import svm

            clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

        else:
            raise ValueError("Unrecognised classifier type {}".format(class_type))

        if return_param_dist:
            return clf, param_dist
        else:
            return clf

    @staticmethod
    def get_cross_val_set(strategy="shuffle", cross_val_params={}):
        if strategy == "shuffle":
            from sklearn.model_selection import StratifiedShuffleSplit

            cross_val_params.setdefault("n_splits", 10)
            cross_val_params.setdefault("test_size", 0.2)
            cross_val_params.setdefault("random_state", 0)
            shuffle = StratifiedShuffleSplit(**cross_val_params)
        else:
            raise ValueError("Unrecognised cross validation {}".format(strategy))
        return shuffle

    def decode(self, class_type="nn"):
        """Decode by fitting with default parameters."""
        # TODO change the test set
        features = self.get_features()
        clf = self.get_classifier()
        to_predict = self.get_labels()
        clf.fit(features[:-10], to_predict[:-10])
        output = clf.predict(features[-10:])
        return clf, output

    @staticmethod
    def decoding_accuracy(true, predicted, target_names=None, as_dict=False):
        """
        A report on decoding accuracy from true and predicted.

        Target names indicates the name of the labels (usually 0, 1, 2...)
        """
        from sklearn.metrics import classification_report

        return classification_report(
            true, predicted, target_names=target_names, output_dict=as_dict
        )

    def cross_val_decoding(
        self,
        class_type="nn",
        classifier_params={},
        cv_strategy="shuffle",
        cross_val_params={"n_splits": 10, "test_size": 0.2, "random_state": 0},
        feature_type="window",
        feature_params={},
        scoring=["accuracy", "balanced_accuracy"],
        verbose=False,
    ):
        """
        Perform decoding with cross-validation.

        """
        from sklearn.model_selection import cross_validate

        clf = self.get_classifier(
            class_type=class_type, classifier_params=classifier_params
        )
        cv = self.get_cross_val_set(
            strategy="shuffle", cross_val_params=cross_val_params
        )
        features = self.get_features(
            feature_type=feature_type, feature_params=feature_params
        )
        labels = self.get_labels()
        print(
            "Running cross val on\n{}\nwith cv\n{}\nusing features: {}".format(
                clf, cv, feature_type
            )
        )
        return cross_validate(
            clf, features, labels, return_train_score=True, scoring=scoring, cv=cv
        )

    def hyper_param_search(
        self,
        n_top=3,
        class_type="nn",
        classifier_params={},
        cv_strategy="shuffle",
        cross_val_params={"n_splits": 10, "test_size": 0.2, "random_state": 0},
        feature_type="window",
        feature_params={},
        scoring=["accuracy", "balanced_accuracy"],
        verbose=False,
    ):
        """
        Perform hyper-param searching.
        """
        from sklearn.model_selection import RandomizedSearchCV

        def report(results, n_top=n_top):
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results["rank_test_score"] == i)
                for candidate in candidates:
                    print("Model with rank: {0}".format(i))
                    print(
                        "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                            results["mean_test_score"][candidate],
                            results["std_test_score"][candidate],
                        )
                    )
                    print("Parameters: {0}".format(results["params"][candidate]))
                    print("")

        clf, param_dist = self.get_classifier(
            class_type=class_type,
            classifier_params=classifier_params,
            return_param_dist=True,
        )
        cv = self.get_cross_val_set(
            strategy="shuffle", cross_val_params=cross_val_params
        )
        features = self.get_features(
            feature_type=feature_type, feature_params=feature_params
        )
        labels = self.get_labels()
        random_search = RandomizedSearchCV(
            clf, param_distributions=param_dist, n_iter=30, cv=cv
        )
        random_search.fit(features, labels)

        if verbose:
            report(random_search.cv_results_)

        # clf.set_params(**random_search.best_params_)
        return random_search

        # You can set the params on a classifier using

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

    # Random one or zero labels for now
    labels = np.random.randint(low=0, high=2, size=100)
    target_names = ["Random OFF", "Random ON"]

    decoder = LFPDecoder(mne_epochs=random_epochs, labels=labels)
    out = decoder.decode()
    print("Actual :", labels[-10:])
    print("Predict:", out[1])
    print(decoder.decoding_accuracy(labels[-10:], out[1], target_names))

    print("----------Cross Validation-------------")
    from pprint import pprint

    cv_result = decoder.cross_val_decoding(
        cross_val_params={"n_splits": 100, "test_size": 0.2, "random_state": 0},
        verbose=True,
    )
    pprint(cv_result)
    pprint(decoder.confidence_interval_estimate(cv_result, "accuracy"))

    random_search = decoder.hyper_param_search(verbose=True)
    print("Best params:", random_search.best_params_)


if __name__ == "__main__":
    random_decoding()
