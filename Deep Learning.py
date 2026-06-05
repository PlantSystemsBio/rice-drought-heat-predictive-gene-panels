###############################
#### Deep Learning Algorithms #####
###############################

# ======= MLPNN ======

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

feature_subsets = {
    "Stability_sel": ['LOC_Os10g38000', 'LOC_Os05g11610', 'LOC_Os01g74140', 'LOC_Os03g13030',
                      'LOC_Os04g40874', 'LOC_Os01g62830', 'LOC_Os04g16130', 'LOC_Os10g41270',
                      'LOC_Os11g13840', 'LOC_Os10g36520'],
    "Boruta":        ['LOC_Os01g04330', 'LOC_Os01g04590', 'LOC_Os01g06590', 'LOC_Os01g12000',
                      'LOC_Os01g14100', 'LOC_Os01g15350', 'LOC_Os01g15640', 'LOC_Os01g18220',
                      'LOC_Os01g19170', 'LOC_Os01g22600'],
    "RFE":           ['LOC_Os01g11280', 'LOC_Os01g74140', 'LOC_Os03g33520', 'LOC_Os03g38210',
                      'LOC_Os03g44290', 'LOC_Os04g16130', 'LOC_Os05g11610', 'LOC_Os09g30250',
                      'LOC_Os10g38000', 'LOC_Os10g41270'],
    "MRMR":          ['LOC_Os03g14010', 'LOC_Os07g42280', 'LOC_Os06g14670', 'LOC_Os01g18220',
                      'LOC_Os08g37444', 'LOC_Os03g09170', 'LOC_Os03g26910', 'LOC_Os09g26920',
                      'LOC_Os04g55920', 'LOC_Os02g52780'],
    "Tree_based":    ['LOC_Os03g09170', 'LOC_Os05g35740', 'LOC_Os06g06880', 'LOC_Os09g23780',
                      'LOC_Os10g07229', 'LOC_Os12g31400', 'LOC_Os11g09230', 'LOC_Os01g12180',
                      'LOC_Os07g15880', 'LOC_Os08g04840'],
    "Genetic":       ['LOC_Os01g51920', 'LOC_Os03g12320', 'LOC_Os04g45810', 'LOC_Os05g32210',
                      'LOC_Os06g51220', 'LOC_Os07g35520', 'LOC_Os07g44610', 'LOC_Os08g35760',
                      'LOC_Os10g30820', 'LOC_Os02g51164']
} ## Drought matrix

feature_subsets = {
    "Stability_sel": ['LOC_Os03g33590', 'LOC_Os02g43410', 'LOC_Os09g37910', 'LOC_Os11g13980'],

    "Boruta":        ['LOC_Os01g01610', 'LOC_Os01g04340', 'LOC_Os01g04360', 'LOC_Os01g04370'],

    "RFE":           ['LOC_Os01g04360', 'LOC_Os03g14180', 'LOC_Os08g25690', 'LOC_Os11g13980'],

    "MRMR":          ['LOC_Os03g14180', 'LOC_Os03g64210', 'LOC_Os04g44830', 'LOC_Os01g04370']
} ## Heat matrix


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


hidden_layer_sizes = []
for n_layers in range(1, 5):  # 1, 2, 3, 4 layers
    for combo in itertools.product([50, 100], repeat=n_layers):
        hidden_layer_sizes.append(combo)

param_grid = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-3, 1e-2],
    'activation': ['relu', 'tanh']
}

mlp_results = []

for name, genes in feature_subsets.items():
    X_tr_sub = X_train[genes].values
    y_tr = y_train.values

    mlp = MLPClassifier(max_iter=1000, random_state=42,  verbose=True)
    grid = GridSearchCV(
        mlp,
        param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True,
        verbose=2
    )

    grid.fit(X_tr_sub, y_tr)

    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    best_mlp = grid.best_estimator_
    y_tr_pred = best_mlp.predict(X_tr_sub)
    train_score = f1_score(y_tr, y_tr_pred, average='macro')

    mlp_results.append({
        'Method': name,
        'Best Params': best_params,
        'Train F1': train_score,
        'CV F1': best_cv_score
    })

df_mlp = pd.DataFrame(mlp_results)

import ace_tools_open as tools; tools.display_dataframe_to_user(name="MLP Hyperparameter Tuning Results", dataframe=df_mlp)


reports = {}

for _, row in df_mlp.iterrows():
    method = row['Method']
    best_params = row['Best Params']
    genes = feature_subsets[method]

    clf = MLPClassifier(
        **best_params,
        max_iter=1000,
        random_state=42
    )

    X_tr_sub = X_train[genes].values
    clf.fit(X_tr_sub, y_train.values)

    y_tr_pred = clf.predict(X_tr_sub)
    X_te_sub = X_test[genes].values
    y_te_pred = clf.predict(X_te_sub)

    cm_train = confusion_matrix(y_train, y_tr_pred, labels=clf.classes_)
    cm_test  = confusion_matrix(y_test, y_te_pred,   labels=clf.classes_)

    cr_train = classification_report(y_train, y_tr_pred, target_names=clf.classes_, digits=3)
    cr_test  = classification_report(y_test,  y_te_pred,  target_names=clf.classes_, digits=3)

    reports[method] = {
        'confusion_matrix_train': cm_train,
        'classification_report_train': cr_train,
        'confusion_matrix_test': cm_test,
        'classification_report_test': cr_test
    }

for method, rep in reports.items():
    print(f"\n==================================================================")
    print(f"\n============================= {method} ===========================")
    print(f"\n==================================================================")
    print("Train Confusion Matrix:")
    print(rep['confusion_matrix_train'])
    print("\nTrain Classification Report:")
    print(rep['classification_report_train'])
    print("\nTest Confusion Matrix:")
    print(rep['confusion_matrix_test'])
    print("\nTest Classification Report:")
    print(rep['classification_report_test'])


##=================== CNN ==========================


import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from scikeras.wrappers import KerasClassifier
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
import numpy as np
import pandas as pd
import ast
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import ast
import numpy as np
import pandas as pd
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score


feature_subsets = {
    "Stability_sel": ['LOC_Os10g38000', 'LOC_Os05g11610', 'LOC_Os01g74140', 'LOC_Os03g13030',
                      'LOC_Os04g40874', 'LOC_Os01g62830', 'LOC_Os04g16130', 'LOC_Os10g41270',
                      'LOC_Os11g13840', 'LOC_Os10g36520'],
    "Boruta":        ['LOC_Os01g04330', 'LOC_Os01g04590', 'LOC_Os01g06590', 'LOC_Os01g12000',
                      'LOC_Os01g14100', 'LOC_Os01g15350', 'LOC_Os01g15640', 'LOC_Os01g18220',
                      'LOC_Os01g19170', 'LOC_Os01g22600'],
    "RFE":           ['LOC_Os01g11280', 'LOC_Os01g74140', 'LOC_Os03g33520', 'LOC_Os03g38210',
                      'LOC_Os03g44290', 'LOC_Os04g16130', 'LOC_Os05g11610', 'LOC_Os09g30250',
                      'LOC_Os10g38000', 'LOC_Os10g41270'],
    "MRMR":          ['LOC_Os03g14010', 'LOC_Os07g42280', 'LOC_Os06g14670', 'LOC_Os01g18220',
                      'LOC_Os08g37444', 'LOC_Os03g09170', 'LOC_Os03g26910', 'LOC_Os09g26920',
                      'LOC_Os04g55920', 'LOC_Os02g52780'],
    "Tree_based":    ['LOC_Os03g09170', 'LOC_Os05g35740', 'LOC_Os06g06880', 'LOC_Os09g23780',
                      'LOC_Os10g07229', 'LOC_Os12g31400', 'LOC_Os11g09230', 'LOC_Os01g12180',
                      'LOC_Os07g15880', 'LOC_Os08g04840'],
    "Genetic":       ['LOC_Os01g51920', 'LOC_Os03g12320', 'LOC_Os04g45810', 'LOC_Os05g32210',
                      'LOC_Os06g51220', 'LOC_Os07g35520', 'LOC_Os07g44610', 'LOC_Os08g35760',
                      'LOC_Os10g30820', 'LOC_Os02g51164']
}

def build_cnn(conv_layers=1, filters=32, kernel_size=2,
              dense_layers=1, dense_units=64,
              dropout_rate=0.3, learning_rate=1e-3):
    model = keras.Sequential()
    # Input reshape for 1D conv
    model.add(keras.layers.Input(shape=(n_features, 1)))
    # Convolutional layers
    for _ in range(conv_layers):
        model.add(keras.layers.Conv1D(filters=filters,
                                      kernel_size=kernel_size,
                                      activation='relu',
                                      padding='same'))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    # Dense layers
    for _ in range(dense_layers):
        model.add(keras.layers.Dense(units=dense_units, activation='relu'))
        model.add(keras.layers.Dropout(rate=dropout_rate))
    # Output
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy'
    )
    return model

cnn_clf = KerasClassifier(model=build_cnn, verbose=1)

param_grid = {
    'model__conv_layers': [1, 2],
    'model__filters':     [16, 32],
    'model__kernel_size': [2, 3],
    'model__dense_layers':[1, 2],
    'model__dense_units': [32, 64],
    'model__dropout_rate':[0.2, 0.4],
    'model__learning_rate':[1e-3, 1e-2],
    'batch_size':         [16, 32],
    'epochs':             [50, 100]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, genes in feature_subsets.items():
    # Prepare data: reshape for CNN
    X_tr_sub = X_train[genes].values
    y_tr = y_train.values
    n_features = X_tr_sub.shape[1]
    X_tr_sub = X_tr_sub.reshape(-1, n_features, 1)

    # Grid search
    grid = HalvingRandomSearchCV(
        cnn_clf,
        param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=1,
        verbose=2,
        refit=True,
        factor=2, min_resources=20, resource='n_samples'
    )

    grid.fit(X_tr_sub, y_tr)

    best_params = grid.best_params_
    best_cv = grid.best_score_

    best_model = grid.best_estimator_
    y_tr_pred = best_model.predict(X_tr_sub)
    train_f1 = f1_score(y_tr, y_tr_pred, average='macro')

    results.append({
        'Method': name,
        'Best Params': best_params,
        'Train F1': train_f1,
        'CV F1': best_cv
    })

df_cnn = pd.DataFrame(results)
import ace_tools_open as tools; tools.display_dataframe_to_user(name="CNN Tuning Results", dataframe=df_cnn)



def build_cnn(conv_layers=1, filters=32, kernel_size=2,
              dense_layers=1, dense_units=64,
              dropout_rate=0.3, learning_rate=1e-3,
              conv_activation='relu', dense_activation='relu'):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features, 1)))
    for _ in range(conv_layers):
        model.add(keras.layers.Conv1D(filters=filters,
                                      kernel_size=kernel_size,
                                      activation=conv_activation,
                                      padding='same'))
        model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Flatten())
    for _ in range(dense_layers):
        model.add(keras.layers.Dense(dense_units,
                                     activation=dense_activation))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy')
    return model

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

activation_results = []

for _, row in df_cnn.iterrows():
    method = row['Method']
    best_params = row['Best Params']
    # parse if string
    if isinstance(best_params, str):
        best_params = ast.literal_eval(best_params)
    genes = feature_subsets[method]

    model_kwargs = {k.replace('model__',''): v
                    for k, v in best_params.items() if k.startswith('model__')}
    fit_kwargs = {k: v for k, v in best_params.items() if not k.startswith('model__')}

    X_tr = X_train[genes].values
    y_tr = y_train.values
    n_features = X_tr.shape[1]
    X_tr = X_tr.reshape(-1, n_features, 1)

    base_clf = KerasClassifier(model=build_cnn, verbose=0, **model_kwargs)
    base_clf.set_params(**fit_kwargs)

    param_grid = {
        'model__conv_activation': ['celu', 'elu', 'exponential', 'gelu', 'glu', 'hard_shrink', 'hard_sigmoid',
    'hard_silu', 'hard_tanh', 'leaky_relu', 'linear', 'log_sigmoid',
    'mish', 'relu', 'selu', 'sigmoid', 'silu', 'softmax', 'softplus',
    'softsign', 'tanh'],

        'model__dense_activation':['celu', 'elu', 'exponential', 'gelu', 'glu', 'hard_shrink', 'hard_sigmoid',
    'hard_silu', 'hard_tanh', 'leaky_relu', 'linear', 'log_sigmoid',
    'mish', 'relu', 'selu', 'sigmoid', 'silu', 'softmax', 'softplus',
    'softsign', 'tanh']
    }

    search = HalvingRandomSearchCV(
        base_clf,
        param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=1,
        verbose=2,
        refit=True,
        factor=2, min_resources=20, resource='n_samples'
    )

    search.fit(X_tr, y_tr)

    best = search.best_params_
    best_conv = best['model__conv_activation']
    best_dense = best['model__dense_activation']
    cv_f1 = search.best_score_

    best_model = search.best_estimator_
    y_tr_pred = best_model.predict(X_tr)
    train_f1 = f1_score(y_tr, y_tr_pred, average='macro')

    activation_results.append({
        'Method': method,
        'Best Conv Activation': best_conv,
        'Best Dense Activation': best_dense,
        'Train F1': train_f1,
        'CV F1': cv_f1
    })

df_activation_tuned = pd.DataFrame(activation_results)
import ace_tools_open as tools; tools.display_dataframe_to_user(
    name="CNN Activation Tuning (Fixed Hyperparams)", dataframe=df_activation_tuned
)

def build_cnn(conv_layers=1, filters=32, kernel_size=2,
              dense_layers=1, dense_units=64,
              dropout_rate=0.3, learning_rate=1e-3,
              conv_activation='relu', dense_activation='relu'):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features, 1)))
    for _ in range(conv_layers):
        model.add(keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=conv_activation,
            padding='same'
        ))
        model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Flatten())
    for _ in range(dense_layers):
        model.add(keras.layers.Dense(
            units=dense_units,
            activation=dense_activation
        ))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy'
    )
    return model

reports_cnn = {}

for _, row in df_cnn.iterrows():
    method = row['Method']
    # parse best hyperparams
    best_params = row['Best Params']
    if isinstance(best_params, str):
        best_params = ast.literal_eval(best_params)

    act_row = df_activation_tuned.loc[df_activation_tuned['Method'] == method].iloc[0]
    conv_act = act_row['Best Conv Activation']
    dense_act = act_row['Best Dense Activation']

    genes = feature_subsets[method]
    X_tr = X_train[genes].values;  X_te = X_test[genes].values
    y_tr = y_train.values;         y_te = y_test.values
    n_features = X_tr.shape[1]
    X_tr = X_tr.reshape(-1, n_features, 1)
    X_te = X_te.reshape(-1, n_features, 1)

    model_kwargs = {}
    fit_kwargs   = {}
    for k, v in best_params.items():
        if k.startswith('model__'):
            model_kwargs[k.replace('model__','')] = v
        else:
            fit_kwargs[k] = v

    model_kwargs['conv_activation']  = conv_act
    model_kwargs['dense_activation'] = dense_act

    clf = KerasClassifier(
        model=build_cnn,
        verbose=0,
        **model_kwargs
    )
    clf.set_params(**fit_kwargs)

    clf.fit(X_tr, y_tr)
    y_tr_pred = clf.predict(X_tr)
    y_te_pred = clf.predict(X_te)

    reports_cnn[method] = {
        'cm_train': confusion_matrix(y_tr, y_tr_pred),
        'cr_train': classification_report(y_tr, y_tr_pred, digits=3),
        'cm_test' : confusion_matrix(y_te, y_te_pred),
        'cr_test' : classification_report(y_te, y_te_pred, digits=3)
    }

for method, rep in reports_cnn.items():
    print(f"\n{'='*60}\n{method:^60}\n{'='*60}")
    print("TRAIN Confusion Matrix:\n", rep['cm_train'])
    print("\nTRAIN Classification Report:\n", rep['cr_train'])
    print("\nTEST Confusion Matrix:\n", rep['cm_test'])
    print("\nTEST Classification Report:\n", rep['cr_test'])


# ============== GAN ================

import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import itertools
import ast
import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

ctgan_grid = {
    'epochs':      [100, 200],
    'batch_size':  [500, 1000]
}

eval_mlp = MLPClassifier(
    hidden_layer_sizes=(50,50),
    alpha=1e-3,
    learning_rate_init=1e-3,
    activation='relu',
    max_iter=1000,
    random_state=42
)
skf = StratifiedKFold(5, shuffle=True, random_state=42)

results = []

for method, genes in feature_subsets.items():
    # prepare the 70% train split
    X_tr = X_train[genes]
    y_tr = y_train
    # isolate minority (Drought) and majority (Control)
    drought_df = X_tr[y_tr=='Drought']
    control_df = X_tr[y_tr=='Control']

    best_score = -np.inf
    best_params = None

    for epochs, bs in itertools.product(ctgan_grid['epochs'], ctgan_grid['batch_size']):
        cv_scores = []

        for train_idx, val_idx in skf.split(X_tr, y_tr):
            X_sub_tr = X_tr.iloc[train_idx]
            y_sub_tr = y_tr.iloc[train_idx]
            X_sub_val= X_tr.iloc[val_idx]
            y_sub_val= y_tr.iloc[val_idx]

            drought_sub = X_sub_tr[y_sub_tr=='Drought'][genes]
            gan = CTGAN(epochs=epochs, batch_size=bs, verbose=False)
            gan.fit(drought_sub)

            control_sub = X_sub_tr[y_sub_tr=='Control'][genes]
            n_ctrl = len(control_sub)
            synth  = gan.sample(n_ctrl)

            X_aug = pd.concat([control_sub, drought_sub, synth], ignore_index=True)
            y_aug = np.array(
                ['Control'] * n_ctrl +
                ['Drought'] * len(drought_sub) +
                ['Drought'] * n_ctrl
            )

            assert X_aug.shape[0] == y_aug.shape[0], "Mismatch X vs y"

            eval_mlp.fit(X_aug, y_aug)
            y_pred = eval_mlp.predict(X_sub_val[genes])
            cv_scores.append(f1_score(y_sub_val, y_pred, average='macro'))

        mean_cv = np.mean(cv_scores)
        if mean_cv > best_score:
            best_score = mean_cv
            best_params = {'epochs': epochs, 'batch_size': bs}

    results.append({
        'Method': method,
        'Best CTGAN Params': best_params,
        'Best CV F1': best_score
    })

df_gan = pd.DataFrame(results)
print(df_gan)


random.seed(42)

final_results = []

sep = "=" * 66

for idx, row in df_gan.iterrows():
    method     = row['Method']
    gan_params = row['Best CTGAN Params']
    genes      = feature_subsets[method]

    print("\n" + sep)
    print(f"{sep}\n============================= {method} ===========================\n{sep}")

    if isinstance(gan_params, str):
        try:
            gan_params = ast.literal_eval(gan_params)
        except Exception:
            print(f"  Warning: could not parse CTGAN params for {method}: {gan_params!r}")
            gan_params = None

    if gan_params is None or (isinstance(gan_params, float) and np.isnan(gan_params)):
        print(f"  Skipping {method}: no valid GAN params found in df_gan.")
        print(sep)
        continue

    X_tr = X_train[genes].reset_index(drop=True)
    y_tr = y_train.reset_index(drop=True)
    X_te = X_test[genes].reset_index(drop=True)
    y_te = y_test.reset_index(drop=True)

    drought_tr = X_tr[y_tr == 'Drought'].reset_index(drop=True)
    control_tr = X_tr[y_tr == 'Control'].reset_index(drop=True)
    n_ctrl     = len(control_tr)

    try:
        if not isinstance(gan_params, dict):
            raise TypeError("gan_params is not a dict after parsing")
        gan = CTGAN(**gan_params, verbose=False)
        gan.fit(drought_tr)
        synth = gan.sample(n_ctrl)

        if isinstance(synth, pd.DataFrame):
            synth = synth[genes] if set(genes).issubset(synth.columns) else synth
        else:
            synth = pd.DataFrame(synth, columns=genes)

    except Exception as e:
        print(f"  Error training/sampling CTGAN for {method}: {e}")
        print("  Skipping this method.")
        print(sep)
        continue

    X_aug = pd.concat([control_tr, drought_tr, synth], ignore_index=True)
    y_aug = np.array(['Control'] * n_ctrl + ['Drought'] * len(drought_tr) + ['Drought'] * n_ctrl)

    try:
        mlp_row = df_mlp.loc[df_mlp['Method'] == method, 'Best Params'].iloc[0]
    except Exception as e:
        print(f"  Could not find MLP params for {method} in df_mlp: {e}")
        print("  Skipping this method.")
        print(sep)
        continue

    best_mlp_params = mlp_row
    if isinstance(best_mlp_params, str):
        try:
            best_mlp_params = ast.literal_eval(best_mlp_params)
        except Exception:
            print(f"  Warning: could not parse MLP params for {method}: {best_mlp_params!r}")
            best_mlp_params = {}

    try:
        clf = MLPClassifier(**best_mlp_params, max_iter=1000, random_state=42)
    except Exception as e:
        print(f"  Error creating MLPClassifier with params {best_mlp_params}: {e}")
        print("  Falling back to default MLPClassifier.")
        clf = MLPClassifier(max_iter=1000, random_state=42)

    try:
        clf.fit(X_aug.values, y_aug)
    except Exception as e:
        print(f"  Error fitting MLP on augmented data for {method}: {e}")
        print("  Skipping this method.")
        print(sep)
        continue

    y_aug_pred = clf.predict(X_aug.values)
    y_te_pred  = clf.predict(X_te.values)

    train_f1 = f1_score(y_aug, y_aug_pred, average='macro')
    test_f1  = f1_score(y_te,  y_te_pred,  average='macro')

    labels = list(clf.classes_)
    cm_train = confusion_matrix(y_aug, y_aug_pred, labels=labels)
    cm_test  = confusion_matrix(y_te,  y_te_pred,  labels=labels)

    cr_train = classification_report(y_aug, y_aug_pred, digits=3)
    cr_test  = classification_report(y_te,  y_te_pred, digits=3)

    print("\nTrain Confusion Matrix:")
    print(cm_train)
    print("\nTrain Classification Report:")
    print(cr_train)

    print("\nTest Confusion Matrix:")
    print(cm_test)
    print("\nTest Classification Report:")
    print(cr_test)

    if cm_test.size == 4 and len(labels) == 2:
        tn, fp, fn, tp = cm_test.ravel()
        print(f"\nLabel order: {labels}")
        print(f"Mapping (Test): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    else:
        print(f"\nLabel order: {labels}")
        print("Note: non-binary or unexpected label ordering; interpret matrix by rows=true, cols=pred.")

    final_results.append({
        'Method':          method,
        'Train F1 (aug)':  train_f1,
        'Test  F1':        test_f1,
        'Train Conf Mat':  cm_train,
        'Test Conf Mat':   cm_test,
        'Train Report':    cr_train,
        'Test Report':     cr_test
    })

    print(sep)

if final_results:
    df_summary = pd.DataFrame([
        {
            'Method': r['Method'],
            'Train F1 (aug)': r['Train F1 (aug)'],
            'Test F1': r['Test  F1']
        } for r in final_results
    ])
    print("\nSummary table:")
    print(df_summary)
else:
    print("\nNo successful results to summarize.")


#============== Deep Autoencoders (DAE) =================

# To find out the progress stage of the codes, tqdm library was used, also the gene panels will be used one by one.

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score
import time
start = time.time()


feature_subsets = {
    "Stability_sel": ['LOC_Os10g38000', 'LOC_Os05g11610', 'LOC_Os01g74140', 'LOC_Os03g13030',
                      'LOC_Os04g40874', 'LOC_Os01g62830', 'LOC_Os04g16130', 'LOC_Os10g41270',
                      'LOC_Os11g13840', 'LOC_Os10g36520']
} ## Other gene panels should insert one by one.

param_grid = {
    'latent_dim':     [2, 5, 10],
    'encoder_layers': [(64,), (128,), (128, 64)],
    'decoder_layers': [(64,), (128,), (64, 128)],
    'activation':     ['relu', 'tanh'],
    'epochs':         [50, 100],
    'batch_size':     [32, 64]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

total_methods = len(feature_subsets)
total_params = len(list(ParameterGrid(param_grid)))
total_steps = total_methods * total_params

method_bar = tqdm(feature_subsets.items(), total=total_methods, desc='Methods')

for method, genes in method_bar:
    X = X_train[genes].values
    y = y_train.values

    best_score = -np.inf
    best_params = None

    for params in tqdm(ParameterGrid(param_grid), total=total_params,
                       desc=f'{method} Params', leave=False):
        cv_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            input_dim = X_tr.shape[1]
            inputs = keras.Input(shape=(input_dim,))
            x = inputs
            for units in params['encoder_layers']:
                x = keras.layers.Dense(units, activation=params['activation'])(x)
            latent = keras.layers.Dense(params['latent_dim'], activation=params['activation'])(x)
            x = latent
            for units in params['decoder_layers']:
                x = keras.layers.Dense(units, activation=params['activation'])(x)
            outputs = keras.layers.Dense(input_dim, activation='linear')(x)

            autoenc = keras.Model(inputs, outputs)
            autoenc.compile(optimizer='adam', loss='mse')
            autoenc.fit(
                X_tr, X_tr,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0
            )

            encoder = keras.Model(inputs, latent)
            X_tr_enc = encoder.predict(X_tr, verbose=0)
            X_val_enc = encoder.predict(X_val, verbose=0)

            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_tr_enc, y_tr)
            y_pred = clf.predict(X_val_enc)
            cv_scores.append(f1_score(y_val, y_pred, average='macro'))

        mean_cv = np.mean(cv_scores)
        if mean_cv > best_score:
            best_score = mean_cv
            best_params = params

    results.append({
        'Method': method,
        'Best Params': best_params,
        'Best CV F1': best_score
    })

df_autoencoder = pd.DataFrame(results)
import ace_tools_open as tools; tools.display_dataframe_to_user(
    name="Autoencoder Tuning Results", dataframe=df_autoencoder
)

end = time.time()
print(f"Run Time: {((end-start)/60):.3f} min")

df_params = pd.read_csv('/content/DAE_Params.csv') # After finding the best parameters for all gene panels, they should be saved in a file.

reports = {}

for _, row in df_params.iterrows():
    method = row['Method']

    params = ast.literal_eval(row['Best Params'])

    activation      = params['activation']
    batch_size      = int(params['batch_size'])
    encoder_layers  = tuple(params['encoder_layers'])
    decoder_layers  = tuple(params['decoder_layers'])
    epochs          = int(params['epochs'])
    latent_dim      = int(params['latent_dim'])

    genes = feature_subsets[method]
    X_tr = X_train[genes].values
    X_te = X_test [genes].values
    y_tr = y_train.values
    y_te = y_test.values

    input_dim = X_tr.shape[1]

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in encoder_layers:
        x = keras.layers.Dense(units, activation=activation)(x)
    latent = keras.layers.Dense(latent_dim, activation=activation)(x)
    x = latent
    for units in decoder_layers:
        x = keras.layers.Dense(units, activation=activation)(x)
    outputs = keras.layers.Dense(input_dim, activation='linear')(x)

    autoenc = keras.Model(inputs, outputs)
    autoenc.compile(optimizer='adam', loss='mse')

    autoenc.fit(
        X_tr, X_tr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    encoder = keras.Model(inputs, latent)
    X_tr_enc = encoder.predict(X_tr, batch_size=batch_size, verbose=0)
    X_te_enc = encoder.predict(X_te, batch_size=batch_size, verbose=0)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_tr_enc, y_tr)

    y_tr_pred = clf.predict(X_tr_enc)
    y_te_pred = clf.predict(X_te_enc)

    cm_train = confusion_matrix(y_tr, y_tr_pred, labels=clf.classes_)
    cm_test  = confusion_matrix(y_te, y_te_pred, labels=clf.classes_)
    cr_train = classification_report(y_tr, y_tr_pred, target_names=clf.classes_, digits=3)
    cr_test  = classification_report(y_te, y_te_pred, target_names=clf.classes_, digits=3)

    reports[method] = {
        'cm_train': cm_train,
        'cr_train': cr_train,
        'cm_test' : cm_test,
        'cr_test' : cr_test
    }

for method, rep in reports.items():
    print(f"\n{'='*60}\n{method:^60}\n{'='*60}")
    print("TRAIN Confusion Matrix:\n", rep['cm_train'])
    print("\nTRAIN Classification Report:\n", rep['cr_train'])
    print("\nTEST Confusion Matrix:\n", rep['cm_test'])
    print("\nTEST Classification Report:\n", rep['cr_test'])














