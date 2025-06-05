Predição Direta de Coordenadas em Redes LoRaWAN Utilizando Random Forest e Comparativo com MLP
Este repositório contém o código-fonte e os dados associados ao estudo comparativo de algoritmos de aprendizado de máquina (Random Forest e Multilayer Perceptron) para a tarefa de localização geográfica de dispositivos em redes LoRaWAN. O trabalho está detalhado no artigo "Predição Direta de Coordenadas em Redes LoRaWAN Utilizando Random Forest Regressor e Comparativo com MLP".

Descrição do Projeto
O projeto investiga uma abordagem baseada em aprendizado de máquina para estimar as coordenadas espaciais (latitude e longitude) de dispositivos LoRaWAN. Utilizam-se características de sinais captados por múltiplos gateways, como RSSI (Received Signal Strength Indicator), SNR (Signal-to-Noise Ratio) e fator de espalhamento (SF).

A metodologia implementada inclui as seguintes etapas principais:

Pré-processamento dos dados: Limpeza e tratamento do dataset original.

Engenharia de Atributos: Criação de novas features a partir dos dados brutos de sinal para enriquecer a informação disponível para os modelos.

Aumento Sintético de Dados: Geração de dados sintéticos com adição de ruído gaussiano para aumentar o volume do conjunto de treinamento, visando melhorar a generalização dos modelos.

Modelagem e Treinamento:

Treinamento de um modelo Random Forest Regressor (composto por um VotingRegressor de dois RFs).

Treinamento de um modelo Multilayer Perceptron (MLP), com otimização de hiperparâmetros via GridSearchCV.

Avaliação e Comparação: Análise do desempenho de ambos os modelos utilizando o Erro Médio Absoluto (MAE) e visualizações gráficas.

Estrutura do Repositório
dataset.csv: O conjunto de dados original utilizado no estudo, contendo medições de sinais LoRaWAN e coordenadas GPS. (Fonte: López Escobar et al., 2024, doi:10.5281/zenodo.13835721).

ruido.py: Script Python responsável pelo carregamento do dataset.csv, aplicação do pré-processamento, engenharia de atributos e aumento sintético de dados. Gera o arquivo dataset_sintetico.csv.

random_forest.py: Script Python que utiliza o dataset_sintetico.csv para treinar, avaliar o modelo Random Forest Regressor e gerar o mapa de predições. Salva os resultados em prediction_results_rf.csv e o mapa em map_direct_rf.html.

mlp.py: Script Python que utiliza o dataset_sintetico.csv para treinar (com GridSearchCV), avaliar o modelo Multilayer Perceptron e gerar o mapa de predições. Salva os resultados em prediction_results_mlp.csv e o mapa em map_direct_mlp.html.

plots.py: Script Python que lê os arquivos prediction_results_rf.csv e prediction_results_mlp.csv para gerar os gráficos individuais e comparativos dos resultados dos modelos. Os gráficos são salvos na pasta plots_gerados/.

plots_gerados/: Diretório contendo todos os gráficos gerados pelo script plots.py. (Este diretório será criado pelo script se não existir).

README.md: Este arquivo.

Requisitos de Software
Para executar os scripts deste projeto, são necessárias as seguintes bibliotecas Python:

pandas

numpy

scikit-learn

matplotlib

seaborn

folium

Você pode instalar todas as dependências de uma vez utilizando o pip:

pip install pandas numpy scikit-learn matplotlib seaborn folium

Instruções de Execução
Para reproduzir os resultados apresentados no artigo, siga os passos abaixo na ordem indicada. Certifique-se de que o arquivo dataset.csv está no mesmo diretório dos scripts.

Gerar o Dataset Sintético:
Execute o script ruido.py para realizar o pré-processamento, engenharia de atributos e aumento de dados. Este script criará o arquivo dataset_sintetico.csv.

python ruido.py

Treinar e Avaliar o Modelo Random Forest:
Execute o script random_forest.py. Ele carregará o dataset_sintetico.csv, treinará o modelo Random Forest e salvará as predições e o mapa.

python random_forest.py

Treinar e Avaliar o Modelo MLP:
Execute o script mlp.py. Ele também carregará o dataset_sintetico.csv, realizará a busca de hiperparâmetros com GridSearchCV, treinará o modelo MLP e salvará as predições e o mapa. Atenção: esta etapa pode ser demorada devido ao GridSearchCV.

python mlp.py

Gerar Gráficos de Resultados:
Após a execução dos scripts dos modelos, execute plots.py para gerar todos os gráficos de desempenho e comparativos. Os gráficos serão salvos na pasta plots_gerados/.

python plots.py

Resultados Esperados
Após a execução dos scripts, os seguintes arquivos principais serão gerados:

dataset_sintetico.csv: Dataset aumentado e pré-processado.

prediction_results_rf.csv: Resultados das predições do modelo Random Forest.

prediction_results_mlp.csv: Resultados das predições do modelo MLP.

map_direct_rf.html: Mapa interativo com as predições do Random Forest.

map_direct_mlp.html: Mapa interativo com as predições do MLP.

Diversos arquivos .png na pasta plots_gerados/ com as visualizações dos resultados (CDFs, histogramas, boxplots, dispersões).

Os resultados de Erro Médio Absoluto (MAE) esperados no conjunto de teste são:

Random Forest: ~24.34 metros

MLP (otimizado): ~156.82 metros

Como Citar este Trabalho
Se você utilizar este código ou os resultados em sua pesquisa, por favor, cite o nosso artigo:

Basso, M. L., Souza, E. G., & Lucca, G. (2025). Predição Direta de Coordenadas em Redes LoRaWAN Utilizando Random Forest Regressor e Comparativo com MLP. 


Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes (se você adicionou um arquivo LICENSE ao criar o repositório).
