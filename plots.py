import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Tamanho uniforme para todas as figuras
FIGSIZE = (7, 5)


def plot_cdf(errors, algorithm_name, output_dir="plots"):
    """Plota a Função de Distribuição Acumulada (CDF) do erro."""
    sns.set_theme(style="whitegrid")  # Aplicar estilo Seaborn aqui
    plt.figure(figsize=FIGSIZE)
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cdf, label=f'CDF - {algorithm_name}')
    plt.xlabel('Erro de Localização (m)')
    plt.ylabel('Fração Acumulada')
    plt.title(f'CDF do Erro de Localização - {algorithm_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cdf_erro_{algorithm_name.lower().replace(" ", "_")}.png'))
    plt.close()


def plot_histogram(errors, algorithm_name, output_dir="plots", x_limit=200):
    """Plota o histograma dos erros."""
    sns.set_theme(style="whitegrid")  # Aplicar estilo Seaborn aqui
    plt.figure(figsize=FIGSIZE)
    plt.hist(errors[errors <= x_limit], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Erro de Localização (m)')
    plt.ylabel('Frequência')
    plt.title(f'Histograma dos Erros ({algorithm_name}, até {x_limit}m)')
    plt.xlim(0, x_limit)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histograma_erro_{algorithm_name.lower().replace(" ", "_")}.png'))
    plt.close()


def plot_scatter_real_vs_predito(df_results, x_true_col, y_true_col, x_pred_col, y_pred_col, algorithm_name,
                                 output_dir="plots"):
    """Plota a dispersão dos pontos reais vs. preditos."""
    sns.set_theme(style="whitegrid")  # Aplicar estilo Seaborn aqui
    plt.figure(figsize=FIGSIZE)
    plt.scatter(df_results[x_true_col], df_results[y_true_col], c='blue', label='Real', alpha=0.6, s=15, edgecolors='w',
                linewidth=0.5)
    plt.scatter(df_results[x_pred_col], df_results[y_pred_col], c='red', label=f'Predito ({algorithm_name})', alpha=0.6,
                s=15, edgecolors='w', linewidth=0.5)
    plt.xlabel('Coordenada X (m)')
    plt.ylabel('Coordenada Y (m)')
    plt.title(f'Dispersão: Real vs. Predito ({algorithm_name})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dispersao_real_vs_predito_{algorithm_name.lower().replace(" ", "_")}.png'))
    plt.close()


def plot_boxplot(errors, algorithm_name, output_dir="plots", zoom_limit=100):
    """Plota o boxplot horizontal dos erros."""
    sns.set_theme(style="whitegrid")  # Aplicar estilo Seaborn aqui
    plt.figure(figsize=FIGSIZE)
    sns.boxplot(x=errors, color='skyblue', orient='h', fliersize=3, linewidth=1.5)
    plt.xlim(0, zoom_limit)
    plt.xlabel('Erro de Localização (m)')
    plt.title(f'Boxplot dos Erros ({algorithm_name}, até {zoom_limit}m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplot_erro_{algorithm_name.lower().replace(" ", "_")}.png'))
    plt.close()


# --- Funções para Gráficos Comparativos ---

def plot_comparative_cdf(errors_rf, errors_mlp, output_dir="plots"):
    """Plota CDFs comparativas para RF e MLP."""
    sns.set_theme(style="whitegrid")  # Aplicar estilo Seaborn aqui
    plt.figure(figsize=FIGSIZE)

    sorted_errors_rf = np.sort(errors_rf)
    cdf_rf = np.arange(1, len(sorted_errors_rf) + 1) / len(sorted_errors_rf)
    plt.plot(sorted_errors_rf, cdf_rf, label='CDF - Random Forest', color='blue', linestyle='-')

    sorted_errors_mlp = np.sort(errors_mlp)
    cdf_mlp = np.arange(1, len(sorted_errors_mlp) + 1) / len(sorted_errors_mlp)
    plt.plot(sorted_errors_mlp, cdf_mlp, label='CDF - MLP', color='green', linestyle='--')

    plt.xlabel('Erro de Localização (m)')
    plt.ylabel('Fração Acumulada')
    plt.title('CDF Comparativa do Erro de Localização')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cdf_comparativa_erro.png'))
    plt.close()


def plot_comparative_boxplot(errors_dict, output_dir="plots", zoom_limit=100):
    """Plota boxplots comparativos dos erros."""
    sns.set_theme(style="whitegrid")  # Aplicar estilo Seaborn aqui
    df_errors = pd.DataFrame(errors_dict)

    plt.figure(figsize=(FIGSIZE[0], FIGSIZE[1] * 0.7 * len(errors_dict)))
    sns.boxplot(data=df_errors, orient='h', palette={'Random Forest': 'skyblue', 'MLP': 'lightgreen'}, fliersize=3,
                linewidth=1.5)
    plt.xlim(0, zoom_limit)
    plt.xlabel('Erro de Localização (m)')
    plt.title(f'Boxplot Comparativo dos Erros (até {zoom_limit}m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_comparativo_erro.png'))
    plt.close()


def plot_comparative_scatter(df_rf, df_mlp, output_dir="plots"):
    """Plota a dispersão comparativa: Real vs. Predito (RF) vs. Predito (MLP)."""
    sns.set_theme(style="whitegrid")  # Aplicar estilo Seaborn aqui
    plt.figure(figsize=(FIGSIZE[0] * 1.2, FIGSIZE[1] * 1.2))

    plt.scatter(df_rf['true_x'], df_rf['true_y'], c='blue', label='Real', alpha=0.5, s=20, edgecolors='w',
                linewidth=0.5)
    plt.scatter(df_rf['pred_x_rf'], df_rf['pred_y_rf'], c='red', label='Predito (RF)', alpha=0.5, s=20, marker='o',
                edgecolors='w', linewidth=0.5)
    plt.scatter(df_mlp['pred_x_mlp'], df_mlp['pred_y_mlp'], c='green', label='Predito (MLP)', alpha=0.5, s=20,
                marker='^', edgecolors='w', linewidth=0.5)

    plt.xlabel('Coordenada X (m)')
    plt.ylabel('Coordenada Y (m)')
    plt.title('Dispersão Comparativa: Real vs. Predições')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dispersao_comparativa_real_vs_predito.png'))
    plt.close()


def main():
    output_plot_dir = "plots_gerados"
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    path_rf = 'prediction_results_rf.csv'
    if os.path.exists(path_rf):
        df_rf = pd.read_csv(path_rf)
        errors_rf = df_rf['error_m_rf'].values
        print(f"Resultados do Random Forest carregados ({len(df_rf)} predições).")

        plot_cdf(errors_rf, "Random Forest", output_plot_dir)
        plot_histogram(errors_rf, "Random Forest", output_plot_dir)
        plot_scatter_real_vs_predito(df_rf, 'true_x', 'true_y', 'pred_x_rf', 'pred_y_rf', "Random Forest",
                                     output_plot_dir)
        plot_boxplot(errors_rf, "Random Forest", output_plot_dir)
        print("Gráficos individuais do Random Forest gerados.")
    else:
        print(f"Arquivo de resultados do Random Forest '{path_rf}' não encontrado.")
        df_rf = None
        errors_rf = np.array([])

    path_mlp = 'prediction_results_mlp.csv'
    if os.path.exists(path_mlp):
        df_mlp = pd.read_csv(path_mlp)
        errors_mlp = df_mlp['error_m_mlp'].values
        print(f"Resultados do MLP carregados ({len(df_mlp)} predições).")

        plot_cdf(errors_mlp, "MLP", output_plot_dir)
        plot_histogram(errors_mlp, "MLP", output_plot_dir)
        plot_scatter_real_vs_predito(df_mlp, 'true_x', 'true_y', 'pred_x_mlp', 'pred_y_mlp', "MLP", output_plot_dir)
        plot_boxplot(errors_mlp, "MLP", output_plot_dir)
        print("Gráficos individuais do MLP gerados.")
    else:
        print(f"Arquivo de resultados do MLP '{path_mlp}' não encontrado.")
        df_mlp = None
        errors_mlp = np.array([])

    if df_rf is not None and df_mlp is not None:
        if len(errors_rf) > 0 and len(errors_mlp) > 0:
            print("Gerando gráficos comparativos...")
            plot_comparative_cdf(errors_rf, errors_mlp, output_plot_dir)

            errors_for_boxplot = {}
            if len(errors_rf) > 0: errors_for_boxplot['Random Forest'] = errors_rf
            if len(errors_mlp) > 0: errors_for_boxplot['MLP'] = errors_mlp
            if errors_for_boxplot:
                plot_comparative_boxplot(errors_for_boxplot, output_plot_dir)

            if 'true_x' in df_rf.columns and 'true_y' in df_rf.columns and \
                    'pred_x_rf' in df_rf.columns and 'pred_y_rf' in df_rf.columns and \
                    'pred_x_mlp' in df_mlp.columns and 'pred_y_mlp' in df_mlp.columns:
                plot_comparative_scatter(df_rf, df_mlp, output_plot_dir)
            else:
                print("Colunas necessárias para scatter comparativo não encontradas em ambos os dataframes.")

            print("Gráficos comparativos gerados.")
        else:
            print("Não há dados de erro suficientes para gerar gráficos comparativos.")
    else:
        print("Não foi possível gerar gráficos comparativos pois faltam resultados de um ou ambos os modelos.")

    print(f"Todos os gráficos foram salvos em '{output_plot_dir}'.")


if __name__ == '__main__':
    main()
