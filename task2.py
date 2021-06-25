import numpy as np
import pickle
import os

from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB

import lightgbm as lgb
from lightgbm import LGBMClassifier

from stacking_ensemble import SklearnWrapper, LightGBMWrapper, StackingEnsemble

from utilities import lgbm_macro_f1, task2_load_cases

lgb_params_emb = {
    'seed': 0,
    'objective': 'binary',
    'verbose': -1,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'num_leaves': 31,
    'feature_fraction': 0.716,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'is_unbalance': 'true'}

lgb_params_textf = {
    'seed': 0,
    'objective': 'binary',
    'verbose': -1,
    'lambda_l1': 6.700219383691919,
    'lambda_l2': 1.001343889145046e-08,
    'num_leaves': 155,
    'feature_fraction': 0.748,
    'bagging_fraction': 0.9714173348977203,
    'bagging_freq': 2,
    'min_child_samples': 20,
    'is_unbalance': 'true'}


def task2_lgbm():
    x_train, y_train, x_val, y_val = task2_load_cases(feature="textf", shuffle=True)
    train_ds = lgb.Dataset(x_train, label=y_train)
    val_ds = lgb.Dataset(x_val, label=y_val)

    model = lgb.train(lgb_params_textf, train_ds, valid_sets=[train_ds, val_ds], feval=lgbm_macro_f1,
                      num_boost_round=10000, early_stopping_rounds=250, verbose_eval=250)

    preds = np.round(model.predict(x_val))
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    with open(f'./saved_models/task1_lgbm_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def task2_stacking_ensemble():
    x_train_emb, y_train, x_val_emb, y_val = task2_load_cases(feature="emb", shuffle=False)
    x_train_textf, _, x_val_textf, _ = task2_load_cases(feature="textf", shuffle=False)

    classifiers_emb = [
        LightGBMWrapper(clf=LGBMClassifier, params=lgb_params_emb),
        SklearnWrapper(clf=RandomForestClassifier()),
        SklearnWrapper(clf=MLPClassifier(max_iter=1000)),
        SklearnWrapper(clf=BernoulliNB())]

    classifiers_textf = [
        LightGBMWrapper(clf=LGBMClassifier, params=lgb_params_textf),
        SklearnWrapper(clf=RandomForestClassifier()),
        SklearnWrapper(clf=MLPClassifier(max_iter=1000)),
        SklearnWrapper(clf=KNeighborsClassifier())]

    ensemble = StackingEnsemble()

    # Training ensemble on embeddings
    ensemble.add_to_ensemble(classifiers_emb, x_train_emb, y_train, x_val_emb, y_val, feature_set_name="emb")

    # Training ensemble on text features
    ensemble.add_to_ensemble(classifiers_textf, x_train_textf, y_train, x_val_textf, y_val, feature_set_name="textf")

    ensemble.train_meta_learner()

    preds = ensemble.predict([x_val_emb, x_val_textf])
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    with open(f'./saved_models/task1_ensemble_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(ensemble, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    task2_lgbm()
    task2_stacking_ensemble()
