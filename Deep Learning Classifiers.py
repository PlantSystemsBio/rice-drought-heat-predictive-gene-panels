#############################################################################################
################################### Deep Learning Classifiers ################################
#############################################################################################

# ========================================== MLPNN ================================================

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


##======================================== CNN ===============================================


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
