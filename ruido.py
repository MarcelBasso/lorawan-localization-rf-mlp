import pandas as pd
import math
import os
import numpy as np

# Constantes (podem ser movidas para um arquivo de configuração ou common.py futuramente)
GATEWAYS = [
    {'id': 1, 'lat': 42.46972, 'lon': -9.01345},
    {'id': 2, 'lat': 42.49955, 'lon': -9.00654},
    {'id': 3, 'lat': 42.50893, 'lon': -9.04902},
]


def latlon_to_xy(lat, lon):
    """Converts latitude and longitude to XY coordinates relative to the first gateway."""
    ref_lat, ref_lon = GATEWAYS[0]['lat'], GATEWAYS[0]['lon']
    x = (lon - ref_lon) * 111320 * math.cos(math.radians(ref_lat))
    y = (lat - ref_lat) * 110540
    return x, y


def load_and_preprocess_original_data(file_path='dataset.csv'):
    """Loads and preprocesses the original dataset."""
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return None

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names

    # Data Cleaning and Filtering (similar to loralocalisation.py)
    df = df[(df['rssi_1'] < 0) & (df['rssi_2'] < 0) & (df['rssi_3'] < 0)]
    df = df.dropna(subset=['rssi_1', 'snr_1', 'rssi_2', 'snr_2', 'rssi_3', 'snr_3', 'spreading_factor', 'lat', 'long'])
    df = df[(df['snr_1'] != 0) & (df['snr_2'] != 0) & (df['snr_3'] != 0)]
    df = df[(df['rssi_1'] > -120) & (df['rssi_2'] > -120) & (df['rssi_3'] > -120)]
    df = df[(df['snr_1'] > -20) & (df['snr_2'] > -20) & (df['snr_3'] > -20)]

    # Feature Engineering (similar to loralocalisation.py)
    df['rssi_mean'] = df[['rssi_1', 'rssi_2', 'rssi_3']].mean(axis=1)
    df['snr_mean'] = df[['snr_1', 'snr_2', 'snr_3']].mean(axis=1)
    df['rssi_std'] = df[['rssi_1', 'rssi_2', 'rssi_3']].std(axis=1)
    df['snr_std'] = df[['snr_1', 'snr_2', 'snr_3']].std(axis=1)
    df['rssi_diff12'] = df['rssi_1'] - df['rssi_2']
    df['rssi_diff13'] = df['rssi_1'] - df['rssi_3']
    df['rssi_diff23'] = df['rssi_2'] - df['rssi_3']
    df['snr_diff12'] = df['snr_1'] - df['snr_2']
    df['snr_diff13'] = df['snr_1'] - df['snr_3']
    df['snr_diff23'] = df['snr_2'] - df['snr_3']
    df['link_1'] = df['rssi_1'] + df['snr_1']
    df['link_2'] = df['rssi_2'] + df['snr_2']
    df['link_3'] = df['rssi_3'] + df['snr_3']
    df['rssi_prod12'] = df['rssi_1'] * df['rssi_2']
    df['rssi_prod13'] = df['rssi_1'] * df['rssi_3']
    df['snr_prod12'] = df['snr_1'] * df['snr_2']
    df['snr_prod13'] = df['snr_1'] * df['snr_3']

    # Handle potential division by zero if any SNR is zero, though filtered earlier
    df['rssi_ratio12'] = np.where(df['rssi_2'] != 0, df['rssi_1'] / df['rssi_2'], np.nan)
    df['rssi_ratio13'] = np.where(df['rssi_3'] != 0, df['rssi_1'] / df['rssi_3'], np.nan)
    df['snr_ratio12'] = np.where(df['snr_2'] != 0, df['snr_1'] / df['snr_2'], np.nan)
    df['snr_ratio13'] = np.where(df['snr_3'] != 0, df['snr_1'] / df['snr_3'], np.nan)
    df = df.dropna()  # Drop rows with NaN created by ratios if any denominator was zero

    # Convert lat/lon to XY and log-transformed XY
    df['x'], df['y'] = zip(*df.apply(lambda r: latlon_to_xy(r['lat'], r['long']), axis=1))
    df['log_x'] = np.log1p(np.abs(df['x'])) * np.sign(df['x'])
    df['log_y'] = np.log1p(np.abs(df['y'])) * np.sign(df['y'])

    print(f"Dados originais válidos carregados e pré-processados: {len(df)} registros de {file_path}")
    return df


def augment_data_with_noise(df, min_points=5000, rssi_noise_std=1.5, snr_noise_std=0.5, lat_lon_noise_m=10.0):
    """Augments the dataset with Gaussian noise to reach a minimum number of points."""
    current_points = len(df)
    if current_points == 0:
        print("Dataset original está vazio. Nenhuma augmentação será feita.")
        return df

    n_copies = max(0, (min_points // current_points) - 1)  # Number of synthetic copies to generate
    if n_copies == 0 and current_points < min_points:  # if we need more points but not enough for a full copy
        n_copies = 1  # ensure at least one copy if below min_points and some augmentation is needed.
    elif n_copies == 0 and current_points >= min_points:
        print(
            f"Número de pontos ({current_points}) já atinge ou excede o mínimo ({min_points}). Nenhuma augmentação de ruído adicional será feita.")
        return df.copy()  # Return a copy to avoid modifying the original df if no augmentation

    print(f"Gerando {n_copies} cópias sintéticas com ruído para tentar atingir ~{min_points} pontos...")

    augmented_dfs = []

    # Store biases for reporting
    rssi_biases, snr_biases, lat_biases, lon_biases = [], [], [], []

    for i in range(n_copies):
        df_aug = df.copy()

        # Add noise to RSSI
        for col in ['rssi_1', 'rssi_2', 'rssi_3']:
            noise = np.random.normal(0, rssi_noise_std, size=len(df_aug))
            df_aug[col] += noise
            rssi_biases.append(np.mean(noise))

        # Add noise to SNR
        for col in ['snr_1', 'snr_2', 'snr_3']:
            noise = np.random.normal(0, snr_noise_std, size=len(df_aug))
            df_aug[col] += noise
            snr_biases.append(np.mean(noise))

        # Add noise to Spreading Factor (categorical, slight perturbation)
        df_aug['spreading_factor'] = df_aug['spreading_factor'].apply(
            lambda x: min(12, max(7, x + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])))
        )

        # Add noise to Latitude and Longitude (convert meters to degrees)
        # Approximate conversion: 1 degree latitude ~ 110.54 km, 1 degree longitude ~ 111.32 km * cos(latitude)
        lat_noise_deg = np.random.normal(0, lat_lon_noise_m / 110540, size=len(df_aug))
        # Using mean latitude for longitude conversion factor for simplicity for the noise
        mean_lat_rad = math.radians(df_aug['lat'].mean() if not df_aug['lat'].empty else GATEWAYS[0]['lat'])
        lon_noise_deg = np.random.normal(0, lat_lon_noise_m / (111320 * math.cos(mean_lat_rad)), size=len(df_aug))

        df_aug['lat'] += lat_noise_deg
        df_aug['long'] += lon_noise_deg
        lat_biases.append(np.mean(lat_noise_deg))
        lon_biases.append(np.mean(lon_noise_deg))

        # Recalculate x, y, log_x, log_y for augmented data
        df_aug['x'], df_aug['y'] = zip(*df_aug.apply(lambda r: latlon_to_xy(r['lat'], r['long']), axis=1))
        df_aug['log_x'] = np.log1p(np.abs(df_aug['x'])) * np.sign(df_aug['x'])
        df_aug['log_y'] = np.log1p(np.abs(df_aug['y'])) * np.sign(df_aug['y'])

        augmented_dfs.append(df_aug)

    if augmented_dfs:
        final_df = pd.concat([df.copy()] + augmented_dfs, ignore_index=True)  # Start with a copy of original df
        print(f"Total de pontos no dataset sintético: {len(final_df)}")
        print("\n--- Relatório de Bias da Augmentação ---")
        if rssi_biases: print(f"Bias médio RSSI: {np.mean(rssi_biases):.4f}")
        if snr_biases: print(f"Bias médio SNR: {np.mean(snr_biases):.4f}")
        if lat_biases: print(f"Bias médio Latitude: {np.mean(lat_biases):.8f} graus")
        if lon_biases: print(f"Bias médio Longitude: {np.mean(lon_biases):.8f} graus")
        print("--------------------------------------\n")
    else:  # No augmentation was done
        final_df = df.copy()

    return final_df


def main():
    """Main function to generate synthetic dataset."""
    print("Iniciando geração de dataset sintético...")
    original_df = load_and_preprocess_original_data(file_path='dataset.csv')

    if original_df is None or original_df.empty:
        print("Não foi possível carregar ou pré-processar os dados originais. Abortando.")
        return

    print(f"Número de pontos originais válidos: {len(original_df)}")

    # Augment data - ensure this function is robust to empty or small df
    synthetic_df = augment_data_with_noise(original_df, min_points=5000)  # Target 5000 points

    if synthetic_df.empty:
        print("Dataset sintético está vazio após a augmentação. Verifique o processo.")
        return

    # Save the synthetic dataset
    output_file = 'dataset_sintetico.csv'
    synthetic_df.to_csv(output_file, index=False)
    print(f"Dataset sintético salvo em: {output_file}")
    print(f"Total de linhas no dataset sintético: {len(synthetic_df)}")


if __name__ == '__main__':
    main()
