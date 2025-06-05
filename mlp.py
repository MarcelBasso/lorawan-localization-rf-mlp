import pandas as pd
import math
import os
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import folium

# Constantes
GATEWAYS = [
    {'id': 1, 'lat': 42.46972, 'lon': -9.01345},
    {'id': 2, 'lat': 42.49955, 'lon': -9.00654},
    {'id': 3, 'lat': 42.50893, 'lon': -9.04902},
]

FEATURE_COLS = [
    'rssi_1', 'snr_1', 'rssi_2', 'snr_2', 'rssi_3', 'snr_3',
    'rssi_mean', 'snr_mean', 'rssi_std', 'snr_std',
    'rssi_diff12', 'rssi_diff13', 'rssi_diff23',
    'snr_diff12', 'snr_diff13', 'snr_diff23',
    'link_1', 'link_2', 'link_3',
    'rssi_prod12', 'rssi_prod13', 'snr_prod12', 'snr_prod13',
    'rssi_ratio12', 'rssi_ratio13', 'snr_ratio12', 'snr_ratio13'
]
CATEGORICAL_COLS = ['spreading_factor']
TARGET_COLS_LOG = ['log_x', 'log_y']


def xy_to_latlon(x, y):
    ref_lat, ref_lon = GATEWAYS[0]['lat'], GATEWAYS[0]['lon']
    lat = y / 110540 + ref_lat
    lon = x / (111320 * math.cos(math.radians(ref_lat))) + ref_lon
    return lat, lon


def build_mlp_model_with_grid_search():
    numerical_features = FEATURE_COLS
    categorical_features = CATEGORICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    mlp_base = MLPRegressor(
        activation='relu',
        solver='adam',
        batch_size='auto',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        verbose=False
    )

    param_grid = {
        'regressor__hidden_layer_sizes': [
            (64, 32),
            (100, 50),
            (128, 64, 32),
            (100, 100),
            (50, 50, 50)
        ],
        'regressor__alpha': [0.0001, 0.001, 0.01],
        'regressor__learning_rate_init': [0.0005, 0.001, 0.005],
    }

    pipeline_for_search = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', mlp_base)])

    print("Configurando GridSearchCV para MLP (otimizando para MAE)...")
    print("Grade de Parâmetros a ser testada:")
    for key, value in param_grid.items():
        print(f"  {key}: {value}")
    print("AVISO: Este processo pode ser demorado!\n")

    # #################################################################### #
    # ## MUDANÇA PRINCIPAL AQUI: de neg_mean_squared_error para neg_mean_absolute_error
    # #################################################################### #
    search = GridSearchCV(
        pipeline_for_search,
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',  # Otimizando para MAE
        n_jobs=-1,
        verbose=2
    )
    return search


def train_and_predict_mlp(df_synthetic, use_grid_search=True):
    missing_cols = [col for col in FEATURE_COLS + CATEGORICAL_COLS if col not in df_synthetic.columns]
    if missing_cols:
        raise ValueError(f"As seguintes colunas de features estão faltando no dataset sintético: {missing_cols}")

    X = df_synthetic[FEATURE_COLS + CATEGORICAL_COLS]
    y_log_x = df_synthetic[TARGET_COLS_LOG[0]]
    y_log_y = df_synthetic[TARGET_COLS_LOG[1]]

    X_train_x, X_test_x, y_train_log_x, y_test_log_x = train_test_split(X, y_log_x, test_size=0.2, random_state=42)
    X_train_y, X_test_y, y_train_log_y, y_test_log_y = train_test_split(X, y_log_y, test_size=0.2, random_state=42)

    print(f"MLP - Treinando modelo para {TARGET_COLS_LOG[0]}...")
    if use_grid_search:
        model_search_x = build_mlp_model_with_grid_search()
        model_search_x.fit(X_train_x, y_train_log_x)
        print(
            f"Melhores parâmetros para {TARGET_COLS_LOG[0]} encontrados pelo GridSearchCV: {model_search_x.best_params_}")
        print(f"Melhor score (neg MAE) para {TARGET_COLS_LOG[0]}: {model_search_x.best_score_}")
        model_log_x = model_search_x.best_estimator_
    else:
        model_log_x = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), FEATURE_COLS),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
                ], remainder='passthrough')),
            ('regressor', MLPRegressor(hidden_layer_sizes=(128, 64, 32), random_state=42, verbose=1))
        ])
        model_log_x.fit(X_train_x, y_train_log_x)

    print(f"\nMLP - Treinando modelo para {TARGET_COLS_LOG[1]}...")
    if use_grid_search:
        model_search_y = build_mlp_model_with_grid_search()
        model_search_y.fit(X_train_y, y_train_log_y)
        print(
            f"Melhores parâmetros para {TARGET_COLS_LOG[1]} encontrados pelo GridSearchCV: {model_search_y.best_params_}")
        print(f"Melhor score (neg MAE) para {TARGET_COLS_LOG[1]}: {model_search_y.best_score_}")
        model_log_y = model_search_y.best_estimator_
    else:
        model_log_y = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), FEATURE_COLS),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
                ], remainder='passthrough')),
            ('regressor', MLPRegressor(hidden_layer_sizes=(128, 64, 32), random_state=42, verbose=1))
        ])
        model_log_y.fit(X_train_y, y_train_log_y)

    pred_log_x = model_log_x.predict(X_test_x)
    pred_log_y = model_log_y.predict(X_test_x)

    pred_x = np.expm1(np.abs(pred_log_x)) * np.sign(pred_log_x)
    pred_y = np.expm1(np.abs(pred_log_y)) * np.sign(pred_log_y)

    predictions_xy = np.vstack((pred_x, pred_y)).T

    return predictions_xy, X_test_x, y_test_log_x, y_test_log_y


def evaluate_predictions_mlp(y_test_log_x_true, y_test_log_y_true, predictions_xy, algorithm_name="MLP"):
    true_x = np.expm1(np.abs(y_test_log_x_true.values)) * np.sign(y_test_log_x_true.values)
    true_y = np.expm1(np.abs(y_test_log_y_true.values)) * np.sign(y_test_log_y_true.values)
    true_xy = np.vstack((true_x, true_y)).T

    errors_m = [np.linalg.norm(t - p) for t, p in zip(true_xy, predictions_xy)]
    mae_m = np.mean(errors_m)
    percentiles = np.percentile(errors_m, [0, 25, 50, 75, 100])

    print(f"\n--- Avaliação Final para {algorithm_name} (após GridSearchCV se usado) ---")
    print(f"MAE XY: {mae_m:.2f} m")
    print(f"Estatísticas do Erro (min, 25%, 50%, 75%, max): {percentiles}")

    total_errors = len(errors_m)
    for threshold in [20, 50, 100, 500, 1000]:
        accuracy = 100 * (np.array(errors_m) <= threshold).sum() / total_errors
        print(f"Accurácia <= {threshold}m: {accuracy:.2f}%")

    df_results = pd.DataFrame({
        'true_x': true_xy[:, 0],
        'true_y': true_xy[:, 1],
        'pred_x_mlp': predictions_xy[:, 0],
        'pred_y_mlp': predictions_xy[:, 1],
        'error_m_mlp': errors_m
    })
    output_csv_path = 'prediction_results_mlp.csv'
    df_results.to_csv(output_csv_path, index=False)
    print(f"Resultados de predição do {algorithm_name} salvos em: {output_csv_path}")
    print("-------------------------------------------------------------------\n")
    return df_results


def generate_map_mlp(df_results):
    m = folium.Map(location=[GATEWAYS[0]['lat'], GATEWAYS[0]['lon']], zoom_start=12)
    for _, row in df_results.iterrows():
        lat_t, lon_t = xy_to_latlon(row['true_x'], row['true_y'])
        lat_p, lon_p = xy_to_latlon(row['pred_x_mlp'], row['pred_y_mlp'])
        folium.CircleMarker([lat_t, lon_t], radius=3, color='blue', fill=True, fill_opacity=0.7, popup="Real").add_to(m)
        folium.CircleMarker([lat_p, lon_p], radius=3, color='purple', fill=True, fill_opacity=0.7,
                            popup="Predito (MLP)").add_to(m)
    for gw in GATEWAYS:
        folium.Marker([gw['lat'], gw['lon']], popup=f"GW{gw['id']}", icon=folium.Icon(color='green')).add_to(m)
    map_file = 'map_direct_mlp.html'
    m.save(map_file)
    print(f"Mapa de predição direta do MLP salvo em: {map_file}")


def main():
    print("Iniciando processo do MLP...")
    synthetic_data_path = 'dataset_sintetico.csv'
    if not os.path.exists(synthetic_data_path):
        print(f"Erro: Dataset sintético '{synthetic_data_path}' não encontrado. Execute ruido.py primeiro.")
        return

    df_synthetic = pd.read_csv(synthetic_data_path)
    print(f"Dataset sintético carregado com {len(df_synthetic)} linhas.")

    predictions_xy, X_test, y_test_log_x, y_test_log_y = train_and_predict_mlp(df_synthetic, use_grid_search=True)

    df_eval_results = evaluate_predictions_mlp(y_test_log_x, y_test_log_y, predictions_xy, algorithm_name="MLP")

    if not df_eval_results.empty:
        generate_map_mlp(df_eval_results)

    print("Processo do MLP concluído.")


if __name__ == '__main__':
    main()

