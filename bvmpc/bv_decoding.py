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

    See cross_val_decoding function for a full pipeline.

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
    label_names : list of str
        The name of each label.
    selected_data : str | list | slice | None
        See mne.Epochs.get_data
        This is the picks argument
    sample_rate : int
        The sampling rate of the lfp data.
    clf : Scipy classifier
        The classifier object to use.
    cv : Scipy cross validation
        The cross validation object to use.
    features : np.ndarray
        Array to use to predict labels
    cross_val_result : dict
        The result of cross validation

    """

    def __init__(
        self,
        mne_epochs=None,
        labels=None,
        label_names=None,
        selected_data=None,
        sample_rate=250,
        clf="nn",
        param_dist=None,
        clf_params={},
        cv=None,
        cv_params={},
        features="window",
        feature_params={},
    ):
        self.mne_epochs = mne_epochs
        self.labels = labels
        self.label_names = label_names
        self.selected_data = selected_data
        self.sample_rate = sample_rate
        self.set_classifier(clf, param_dist, clf_params)
        self.set_cross_val_set(cv, cv_params)
        self.set_features(features, feature_params)
        self.feature_params = feature_params
        self.cross_val_result = None

    def get_labels(self):
        """Return the labels of each epoch as a numpy array."""
        return self.labels

    def get_data(self):
        """
        Return a 3D numpy array of the LFP data.

        This array is in the shape (epochs, chans, times).

        """
        return self.mne_epochs.get_data(picks=self.selected_data)

    def get_classifier(self):
        """Return the classifier used for decoding."""
        return self.clf

    def set_classifier(self, clf, param_dist, clf_params={}):
        """Either set or make a classifier."""
        if isinstance(clf, str):
            clf, param_dist = make_classifier(clf, clf_params, return_param_dist=True)
        self.clf = clf
        self.param_dist = param_dist

    def set_features(self, features, feature_params={}):
        """Features can be either an array or a string, in which case it is made."""
        if features == "window":
            features = window_features(self.get_data(), **feature_params)
        elif isinstance(features, np.ndarray):
            if features.shape[0] != len(self.labels):
                raise ValueError(
                    "features don't match labels in length {}:{}".format(
                        len(features), len(self.labels)
                    )
                )
        else:
            raise ValueError("Unrecognised feature type {}".format(feature_type))
        self.features = features

    def get_features(self):
        """Return the features used for decoding."""
        return self.features

    def set_cross_val_set(self, cv, cv_params={}):
        """Set the cross validation set to be used or make one."""
        if isinstance(cv, str):
            cv = make_cross_val_set(cv, cv_params)
        self.cv = cv

    def get_cross_val_set(self):
        """Get the cross validation set to be used."""
        return self.cv

    def set_all_data(self, mne_epochs, labels, label_names, selected_data=None):
        self.mne_epochs = mne_epochs
        self.labels = labels
        self.label_names = label_names
        self.selected_data = selected_data

    def decode(self, test_size=0.2):
        """
        Decode by fitting with default parameters.

        Parameters
        ----------
        test_size : float
            The ratio of the random test set to use for decoding.

        Returns
        -------
        (clf, output, test_labels)

        """
        from sklearn.model_selection import train_test_split

        features = self.get_features()
        clf = self.get_classifier()
        to_predict = self.get_labels()

        train_features, test_features, train_labels, test_labels = train_test_split(
            features, to_predict, test_size=test_size, shuffle=True
        )
        clf.fit(train_features, train_labels)
        output = clf.predict(test_features)
        return clf, output, test_labels

    def cross_val_decode(self, scoring=["accuracy", "balanced_accuracy"]):
        """
        Perform decoding with cross-validation.

        Parameters
        ----------
        scoring : list of strings
            Scikit compatible scoring function names.

        Returns
        -------
        dict

        """
        from sklearn.model_selection import cross_validate

        clf = self.get_classifier()
        cv = self.get_cross_val_set()
        features = self.get_features()
        labels = self.get_labels()
        print("Running cross val on {} with cv {}".format(clf, cv))
        result = cross_validate(
            clf, features, labels, return_train_score=True, scoring=scoring, cv=cv
        )
        self.cross_val_result = result
        return result

    def hyper_param_search(
        self,
        n_top=3,
        scoring=["accuracy", "balanced_accuracy"],
        set_params=True,
        verbose=False,
    ):
        """
        Perform hyper-param searching.

        Parameters
        ----------
        n_top : int, optional. 
            Defaults to 3.
            The number of top parameter results to return.
        scoring : list of str, optional. 
            Defaults to ["accuracy", "balanced_accuracy"]
            Sklearn compatible list of function names.
        set_params : bool, optional.
            Defaults to True.
            Whether to set the best parameters found on the classifier.
        verbose : bool, optional.
            Defaults to False.
            Whether to print extra information about the search.

        Returns
        -------
        dict

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

        clf = self.get_classifier()
        param_dist = self.param_dist
        cv = self.get_cross_val_set()
        features = self.get_features()
        labels = self.get_labels()

        random_search = RandomizedSearchCV(
            clf, param_distributions=param_dist, n_iter=30, cv=cv
        )
        random_search.fit(features, labels)

        if verbose:
            report(random_search.cv_results_)

        if set_params:
            self.clf.set_params(**random_search.best_params_)
        return random_search

    def decoding_accuracy(self, true, predicted, as_dict=False):
        """
        A report on decoding accuracy from true and predicted.

        Target names indicates the name of the labels (usually 0, 1, 2...)
        """
        from sklearn.metrics import classification_report

        labels = []
        for val in self.labels:
            if val not in labels:
                labels.append(val)

        print("Actual   :", true)
        print("Predicted:", predicted)
        return classification_report(
            true,
            predicted,
            labels=labels,
            target_names=self.label_names,
            output_dict=as_dict,
        )

    def confidence_interval_estimate(self, key):
        return confidence_interval_estimate(self.cross_val_result, key)


def confidence_interval_estimate(cross_val_result, key):
    """Returns 95% confidence interval estimates from cross_val results."""
    test_key = "test_" + key
    train_key = "train_" + key
    test_scores = cross_val_result[test_key]
    train_scores = cross_val_result[train_key]

    test_str = "Test {}: {:.2f} (+/- {:.2f})".format(
        key, test_scores.mean(), test_scores.std() * 1.96
    )
    train_str = "Train {}: {:.2f} (+/- {:.2f})".format(
        key, train_scores.mean(), train_scores.std() * 1.96
    )
    return test_str, train_str


def make_classifier(class_type="nn", classifier_params={}, return_param_dist=False):
    """
    Get a classifier matching class_type and pass classifier_params to it.

    If return_param_dist is True, also returns a sensible distribution of
    hyperparameters to search over for that classifier.

    """
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


def make_cross_val_set(strategy="shuffle", cross_val_params={}):
    """Get a split of the data into cross validation sets."""
    if strategy == "shuffle":
        from sklearn.model_selection import StratifiedShuffleSplit

        cross_val_params.setdefault("n_splits", 10)
        cross_val_params.setdefault("test_size", 0.2)
        cross_val_params.setdefault("random_state", 0)
        shuffle = StratifiedShuffleSplit(**cross_val_params)
    else:
        raise ValueError("Unrecognised cross validation {}".format(strategy))
    return shuffle


def window_features(data, window_sample_len=10, step=8):
    """Compute features from LFP in windows."""
    from skimage.util import view_as_windows

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


def random_decoding():
    """Perform a full decoding pipeline from random white noise."""
    from bvmpc.bv_mne import random_white_noise
    from pprint import pprint

    # Just random white noise signal
    random_epochs = random_white_noise(100, 4, 500)

    # Random one or zero labels for now
    labels = np.random.randint(low=0, high=2, size=100)
    target_names = ["Random OFF", "Random ON"]

    decoder = LFPDecoder(
        mne_epochs=random_epochs, labels=labels, label_names=target_names
    )
    out = decoder.decode()
    print(decoder.decoding_accuracy(out[2], out[1]))

    print("\n----------Cross Validation-------------")

    decoder.cross_val_decode(verbose=True)
    pprint(decoder.cross_val_result)
    pprint(decoder.confidence_interval_estimate("accuracy"))

    random_search = decoder.hyper_param_search(verbose=True, set_params=False)
    print("Best params:", random_search.best_params_)


if __name__ == "__main__":
    random_decoding()
