import copy
import numpy as np
import pandas as pd
import os
import contextlib

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
NFOLDS = 4
KFOLD = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)


def skl_macro_f1(y_true, y_hat):
    """Early stopping by macro F1-score, callback function for LightGBM sklearn API."""
    y_hat = np.where(y_hat > 0.5, 1, 0)
    return 'f1', f1_score(y_true, y_hat, average='macro'), True


class SklearnWrapper(object):
    """Wapper object for Sklearn classifiers."""

    def __init__(self, clf, seed=SEED, params=None, scale=False):

        if scale:
            if params is None:
                self.clf = make_pipeline(StandardScaler(), clf)
            else:
                self.clf = make_pipeline(StandardScaler(), clf(**params))
        else:
            if params is None:
                self.clf = clf
            else:
                self.clf = clf(**params)

        self.clftype = type(clf)

    def train(self, x_train, y_train, x_val=None, y_val=None):
        self.clf.fit(X=x_train, y=y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]

    def __str__(self):
        return str(self.clftype).split(".")[-1][:-2]


class LightGBMWrapper(object):
    """Wrapper object for LightGBMClassifier."""

    def __init__(self, clf, seed=SEED, params=None):
        params['feature_fraction_seed'] = seed
        params['bagging_seed'] = seed
        self.params = params
        self.clf = clf(**params, n_estimators=10000)

    def train(self, x_train, y_train, x_val, y_val):
        self.clf.fit(X=x_train, y=y_train, eval_set=(x_val, y_val), verbose=0, early_stopping_rounds=250,
                     eval_metric=skl_macro_f1)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]

    def __str__(self):
        return str(type(self.clf)).split(".")[-1][:-2]


def get_oof(clf, x_train, y_train, x_test, y_test):
    """Get stacked out-of-fold predictions on training data and save classifiers
    for future predictions."""

    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, x_test.shape[0]))
    models = []

    for i, (train_index, val_index) in enumerate(KFOLD.split(x_train, y_train)):

        x_train_fold = x_train[train_index, :]
        y_train_fold = y_train[train_index]
        x_val_fold = x_train[val_index, :]
        y_val_fold = y_train[val_index]

        clf.train(x_train_fold, y_train_fold, x_val_fold, y_val_fold)

        train_pred = clf.predict(x_train_fold)
        oof_pred = clf.predict(x_val_fold)
        test_pred = clf.predict(x_test)

        oof_train[val_index] = oof_pred
        oof_test_skf[i, :] = test_pred

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            models.append(copy.deepcopy(clf))

        train_f1 = f1_score(y_train_fold, np.round(train_pred), average='macro')
        val_f1 = f1_score(y_val_fold, np.round(oof_pred), average='macro')
        test_f1 = f1_score(y_test, np.round(test_pred), average='macro')

        print(f'Fold {i + 1}/{NFOLDS}, {clf}, train macro-F1: {train_f1:.3f}, oof macro-F1: {val_f1:.3f}, '
              f'macro-F1: {test_f1:.3f}')

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1).ravel(), oof_test.reshape(-1, 1).ravel(), models


class StackingEnsemble:
    """Stacking ensemble classifier.

    To add classifiers, call 'add_to_ensemble' and provide a list of wrappers, a training set for oof predictions,
    and test set for validation. The feature set needs a name when training parts of the ensemble on different sets.

    After adding classifiers, 'train_meta_learner' needs to be called to train on out-of-fold training predictions.

    Predictions can be made on new data provided a list of the same features that was used during training classifiers.
    """

    def __init__(self):

        self.initialised = False
        self.ready_for_meta_learning = False
        self.oof_train = pd.DataFrame()
        self.oof_test = pd.DataFrame()
        self.y_train = None
        self.y_test = None

        self.clf_count = 0
        self.feature_set_count = 0
        self.clf_feature_set_ids = []
        self.feature_sets = dict()
        self.models = []
        self.metalearner = None

    def add_to_ensemble(self, clf_wrapper_list, x_train, y_train, x_test, y_test, feature_set_name):
        """Train classifiers on provided feature set, add and save to ensemble object."""

        print(f"\nAdding to ensemble, {len(clf_wrapper_list)} classifiers trained on input {x_train.shape}:\n")

        if feature_set_name in self.feature_sets:
            feature_set_id = self.feature_sets['feature_set_name']
        else:
            feature_set_id = self.feature_set_count
            self.feature_sets['feature_set_name'] = self.feature_set_count
            self.feature_set_count += 1

        if self.initialised:
            assert (self.y_train == y_train).all() and (self.y_test == y_test).all(), "provided dataset is different to previously fitted set"
        else:
            self.initialised = True
            self.y_train = y_train
            self.y_test = y_test

        for clf in clf_wrapper_list:
            oof_train, oof_test, models = get_oof(clf, x_train, y_train, x_test, y_test)
            self.oof_train[f'{self.feature_set_count}_{self.clf_count}'] = oof_train
            self.oof_test[f'{self.feature_set_count}_{self.clf_count}'] = oof_test
            self.models.append(models)
            self.clf_count += 1
            self.clf_feature_set_ids.append(feature_set_id)

        self.ready_for_meta_learning = True

    def train_meta_learner(self):
        """Train meta-learner on out-of-fold predictions.

        Can only be called after having called 'add_to_ensemble'."""

        assert self.ready_for_meta_learning is True

        print(f"\nTraining meta-learner on ensemble of {self.clf_count} classifiers:")
        self.metalearner = LogisticRegression()
        self.metalearner.fit(self.oof_train, self.y_train)

        preds = self.metalearner.predict(self.oof_train)
        ac = accuracy_score(self.y_train, preds)
        f1 = f1_score(self.y_train, preds, average='macro')
        print(f"Train: accuracy {ac:0.3f}, macro-F1 {f1:0.3f}")

        preds = self.metalearner.predict(self.oof_test)
        ac = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average='macro')
        print(f"Valid: accuracy {ac:0.3f}, macro-F1 {f1:0.3f} ")

    def predict_proba(self, fs_list):
        """Predict probabilities on a list of feature sets, the same used when training the ensemble."""

        assert self.metalearner is not None
        basepreds = pd.DataFrame()

        for i, clf_models in enumerate(self.models):

            fs_id = self.clf_feature_set_ids[i]

            clf_preds = np.zeros((fs_list[fs_id].shape[0],))
            preds_skf = np.empty((NFOLDS, fs_list[fs_id].shape[0]))

            for j, clf in enumerate(clf_models):
                pred = clf.predict(fs_list[fs_id])
                preds_skf[j, :] = pred

            clf_preds[:] = preds_skf.mean(axis=0)
            basepreds[i] = clf_preds

        preds_prob = self.metalearner.predict_proba(basepreds)[:, 1]
        return preds_prob

    def predict(self, fs_list):
        """Predict binary classes for a list of feature sets, the same used when training the ensemble."""

        assert self.metalearner is not None
        basepreds = pd.DataFrame()

        for i, clf_models in enumerate(self.models):

            fs_id = self.clf_feature_set_ids[i]

            clf_preds = np.zeros((fs_list[fs_id].shape[0],))
            preds_skf = np.empty((NFOLDS, fs_list[fs_id].shape[0]))

            for j, clf in enumerate(clf_models):
                pred = clf.predict(fs_list[fs_id])
                preds_skf[j, :] = pred

            clf_preds[:] = preds_skf.mean(axis=0)
            basepreds[i] = clf_preds

        preds = self.metalearner.predict(basepreds)
        return preds

    def evaluate(self, fs_list, y):
        """Evaluate ensemble given a list of feature sets and labels."""

        preds = self.predict(fs_list)
        ac = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average='macro')
        print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}")
