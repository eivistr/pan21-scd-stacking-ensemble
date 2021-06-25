import time
import pickle

import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna

from utilities import task1_load_cases, task2_load_cases, task3_load_cases, lgbm_macro_f1


def tune_lgbm(x_train, y_train, x_test, y_test, save_as):

    train_ds = lgb.Dataset(x_train, label=y_train)
    val_ds = lgb.Dataset(x_test, label=y_test)

    opt_params = {
        "seed": 0,
        "objective": "binary",
        "boosting_type": "gbdt",
        "verbose": -1
    }

    optuna_model = lgb_optuna.train(opt_params, train_ds, valid_sets=val_ds, feval=lgb_f1_score,
                                    num_boost_round=2500, early_stopping_rounds=100, verbose_eval=-1)

    # Save results
    with open('./optuna/' + save_as, 'wb') as handle:
        pickle.dump(optuna_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def opt_task1():
    x_train, y_train, x_val, y_val = task1_load_cases(feature="emb", shuffle=True)
    tune_lgbm(x_train, y_train, x_val, y_val, save_as="opt_lgbm_t1_emb.pickle")

    x_train, y_train, x_val, y_val = task1_load_cases(feature="textf", shuffle=True)
    tune_lgbm(x_train, y_train, x_val, y_val, save_as="opt_lgbm_t1_textf.pickle")


def opt_task2():
    x_train, y_train, x_val, y_val = task2_load_cases(feature="emb", shuffle=True)
    tune_lgbm(x_train, y_train, x_val, y_val, save_as="opt_lgbm_t2_emb.pickle")

    x_train, y_train, x_val, y_val = task2_load_cases(feature="textf", shuffle=True)
    tune_lgbm(x_train, y_train, x_val, y_val, save_as="opt_lgbm_t2_textf.pickle")


def opt_task3():
    x_train, y_train, x_val, y_val = task3_load_cases(feature="emb", shuffle=True)
    tune_lgbm(x_train, y_train, x_val, y_val, save_as="opt_lgbm_t3_emb.pickle")

    x_train, y_train, x_val, y_val = task3_load_cases(feature="textf", shuffle=True)
    tune_lgbm(x_train, y_train, x_val, y_val, save_as="opt_lgbm_t3_textf.pickle")


if __name__ == '__main__':
    opt_task1()
    opt_task2()
    opt_task3()
