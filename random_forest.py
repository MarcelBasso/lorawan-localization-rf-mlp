import pandas as pd
import math
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import folium  # For map generation

# Constantes (deveriam idealmente vir de um common.py ou config.py)
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
TARGET_COLS_LOG = ['log_x', 'log_y']  # We predict log-transformed coordinates


def xy_to_latlon(x, y):
    """Converts XY coordinates back to latitude and longitude."""
    ref_lat, ref_lon = GATEWAYS[0]['lat'], GATEWAYS[0]['lon']
    lat = y / 110540 + ref_lat
    lon = x / (111320 * math.cos(math.radians(ref_lat))) + ref_lon
    return lat, lon


def build_rf_model():
    """Builds the Random Forest model pipeline with preprocessor."""
    # Define numerical and categorical features based on the synthetic dataset
    # Note: 'spreading_factor' is the only categorical feature used in original script

    numerical_features = FEATURE_COLS
    categorical_features = CATEGORICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'  # Keep other columns if any, though not expected for X
    )

    # Ensemble of RandomForestRegressors as in the original script
    rf1 = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)
    rf2 = RandomForestRegressor(n_estimators=700, max_depth=30, random_state=52, n_jobs=-1)

    # Create the VotingRegressor
    ensemble_model = VotingRegressor([('rf1', rf1), ('rf2', rf2)])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ensemble_model)
    ])
    return model_pipeline


def train_and_predict_rf(df_synthetic):
    """Trains Random Forest models for log_x and log_y, and makes predictions."""

    # Define features (X) and targets (y)
    # Ensure all feature columns are present
    missing_cols = [col for col in FEATURE_COLS + CATEGORICAL_COLS if col not in df_synthetic.columns]
    if missing_cols:
        raise ValueError(f"As seguintes colunas de features estão faltando no dataset sintético: {missing_cols}")

    X = df_synthetic[FEATURE_COLS + CATEGORICAL_COLS]
    y_log_x = df_synthetic[TARGET_COLS_LOG[0]]
    y_log_y = df_synthetic[TARGET_COLS_LOG[1]]

    # Split data
    X_train_x, X_test_x, y_train_log_x, y_test_log_x = train_test_split(X, y_log_x, test_size=0.2, random_state=42)
    # For y_log_y, we use the same X_train and X_test indices for consistency
    X_train_y, X_test_y, y_train_log_y, y_test_log_y = train_test_split(X, y_log_y, test_size=0.2,
                                                                        random_state=42)  # Same split for y

    print(f"Random Forest - Treinando modelo para {TARGET_COLS_LOG[0]}...")
    model_log_x = build_rf_model()
    model_log_x.fit(X_train_x, y_train_log_x)

    print(f"Random Forest - Treinando modelo para {TARGET_COLS_LOG[1]}...")
    model_log_y = build_rf_model()
    model_log_y.fit(X_train_y, y_train_log_y)  # Use X_train_y for y_log_y

    # Predictions on the test set
    # Important: Use the same test set for both x and y predictions.
    # X_test_x and X_test_y should be identical if split with the same random_state.
    # We'll use X_test_x as the common test set.
    pred_log_x = model_log_x.predict(X_test_x)
    pred_log_y = model_log_y.predict(X_test_x)  # Predict on the same X_test for y

    # Inverse transform log predictions to original XY scale
    pred_x = np.expm1(np.abs(pred_log_x)) * np.sign(pred_log_x)
    pred_y = np.expm1(np.abs(pred_log_y)) * np.sign(pred_log_y)

    predictions_xy = np.vstack((pred_x, pred_y)).T

    return predictions_xy, X_test_x, y_test_log_x, y_test_log_y  # Return common X_test and respective y_test


def evaluate_predictions(y_test_log_x_true, y_test_log_y_true, predictions_xy, algorithm_name="Random Forest"):
    """Evaluates predictions and prints metrics, saves results to CSV."""
    # Inverse transform true log_x and log_y to compare in original XY scale
    true_x = np.expm1(np.abs(y_test_log_x_true.values)) * np.sign(y_test_log_x_true.values)
    true_y = np.expm1(np.abs(y_test_log_y_true.values)) * np.sign(y_test_log_y_true.values)

    true_xy = np.vstack((true_x, true_y)).T

    # Calculate Euclidean errors in meters
    errors_m = [np.linalg.norm(t - p) for t, p in zip(true_xy, predictions_xy)]

    mae_m = np.mean(errors_m)
    percentiles = np.percentile(errors_m, [0, 25, 50, 75, 100])

    print(f"\n--- Avaliação Final para {algorithm_name} ---")
    print(f"MAE XY: {mae_m:.2f} m")
    print(f"Estatísticas do Erro (min, 25%, 50%, 75%, max): {percentiles}")

    total_errors = len(errors_m)
    for threshold in [20, 50, 100, 500, 1000]:
        accuracy = 100 * (np.array(errors_m) <= threshold).sum() / total_errors
        print(f"Accurácia <= {threshold}m: {accuracy:.2f}%")

    # Save results to CSV
    df_results = pd.DataFrame({
        'true_x': true_xy[:, 0],
        'true_y': true_xy[:, 1],
        'pred_x_rf': predictions_xy[:, 0],  # Suffix for RF
        'pred_y_rf': predictions_xy[:, 1],  # Suffix for RF
        'error_m_rf': errors_m  # Suffix for RF
    })
    output_csv_path = 'prediction_results_rf.csv'
    df_results.to_csv(output_csv_path, index=False)
    print(f"Resultados de predição do {algorithm_name} salvos em: {output_csv_path}")
    print("-------------------------------------------\n")
    return df_results  # Return for potential map generation


def generate_map_rf(df_results):
    """Generates a Folium map for Random Forest predictions."""
    m = folium.Map(location=[GATEWAYS[0]['lat'], GATEWAYS[0]['lon']], zoom_start=12)

    for _, row in df_results.iterrows():
        lat_t, lon_t = xy_to_latlon(row['true_x'], row['true_y'])
        lat_p, lon_p = xy_to_latlon(row['pred_x_rf'], row['pred_y_rf'])

        folium.CircleMarker([lat_t, lon_t], radius=3, color='blue', fill=True, fill_opacity=0.7, popup="Real").add_to(m)
        folium.CircleMarker([lat_p, lon_p], radius=3, color='red', fill=True, fill_opacity=0.7,
                            popup="Predito (RF)").add_to(m)
        # Line between real and predicted
        # folium.PolyLine([(lat_t, lon_t), (lat_p, lon_p)], color="gray", weight=1, opacity=0.5).add_to(m)

    for gw in GATEWAYS:
        folium.Marker([gw['lat'], gw['lon']], popup=f"GW{gw['id']}", icon=folium.Icon(color='green')).add_to(m)

    map_file = 'map_direct_rf.html'
    m.save(map_file)
    print(f"Mapa de predição direta do Random Forest salvo em: {map_file}")


def main():
    """Main function to train RF, predict, and evaluate."""
    print("Iniciando processo do Random Forest...")
    synthetic_data_path = 'dataset_sintetico.csv'
    if not os.path.exists(synthetic_data_path):
        print(f"Erro: Dataset sintético '{synthetic_data_path}' não encontrado. Execute ruido.py primeiro.")
        return

    df_synthetic = pd.read_csv(synthetic_data_path)
    print(f"Dataset sintético carregado com {len(df_synthetic)} linhas.")

    predictions_xy, X_test, y_test_log_x, y_test_log_y = train_and_predict_rf(df_synthetic)

    # Evaluate
    df_eval_results = evaluate_predictions(y_test_log_x, y_test_log_y, predictions_xy, algorithm_name="Random Forest")

    # Generate map
    if not df_eval_results.empty:
        generate_map_rf(df_eval_results)

    print("Processo do Random Forest concluído.")


if __name__ == '__main__':
    main()
