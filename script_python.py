import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#from sklearn.datasets import load_iris
import time
import torch
import os 
from itertools import permutations
import matplotlib.pyplot as plt
os.environ['LD_LIBRARY_PATH'] = '/home/ibpad/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import nvidia.cublas.lib
import nvidia.cudnn.lib
from datetime import datetime


# Definindo o dispositivo de computação
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
np.random.seed(42)
torch.manual_seed(42)

print('Iniciando leitura da base de dados')
base=pd.read_csv('/home/ibpad/mestrado_raquel/base_har70plus_versao_final_25_porcento.csv')
print(f'Leitura da base de dados finalizada, cujo tamanho é {len(base)}')

X = base.drop(['código','Unnamed: 0','timestamp','id','id_teste','data_formatada','hora_completa','hora','minuto','segundo'], axis=1).values
y = base['código'].values.reshape(-1, 1)

print('Arrays formadas')

print('Definindo conjuntos de treino e teste')

# Codificação One-hot das etiquetas
codificador = OneHotEncoder(sparse=False)
y_onehot = codificador.fit_transform(y)

# Dividir os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Converter os arrays numpy para tensores do PyTorch
X_treino, X_teste, y_treino, y_teste = map(lambda x: torch.tensor(x).float().to(device), (X_treino, X_teste, y_treino, y_teste))
#X_treino, X_teste = X_treino.float(), X_teste.float()
#y_treino, y_teste = y_treino.float(), y_teste.float()

print('Iniciando definição de funções')
# Classe do modelo de rede neural
class RNA(nn.Module):
    def __init__(self, tamanhos_camadas, funcoes_ativacao):
        super(RNA, self).__init__()
        self.camadas = nn.ModuleList()
        for i in range(len(tamanhos_camadas) - 1):
            self.camadas.append(nn.Linear(tamanhos_camadas[i], tamanhos_camadas[i+1]))
        self.funcoes_ativacao = funcoes_ativacao

    def forward(self, x):
        for i, camada in enumerate(self.camadas):
            x = camada(x)
            if i < len(self.camadas) - 1:
                x = self.funcoes_ativacao[i](x)
        return x

# Função para treinar um modelo de rede neural
def treinar_modelo(modelo, X_treino, y_treino, taxa_aprendizado, tamanho_lote, escolha_otimizador,
                   momento=None, criterio_parada='loss', limiar_parada=0.1, max_epocas=500):
    modelo = modelo.to(device)
    # Preparação dos dados para treinamento
    dados_treino = TensorDataset(X_treino, y_treino)
    print(f'len dados_treino:{len(dados_treino)}')
    carregador_treino = DataLoader(dataset=dados_treino, batch_size=tamanho_lote, shuffle=True)
    print(f'len carregador_treino:{len(carregador_treino)}')
    # Configuração do otimizador
    if escolha_otimizador == 'SGD':
        if momento is not None:
            otimizador = optim.SGD(modelo.parameters(), lr=taxa_aprendizado, momentum=momento)
        else:
            otimizador = optim.SGD(modelo.parameters(), lr=taxa_aprendizado)
    elif escolha_otimizador == 'Adam':
        otimizador = optim.Adam(modelo.parameters(), lr=taxa_aprendizado)
    else:
        raise ValueError("Otimizador não suportado")

    criterio = nn.CrossEntropyLoss()

    # Dicionário para armazenar os dados do treinamento
    dados_treinamento = {'epoca': [], 'perda': [], 'acuracia': [], 'duracao_epoca': []}

    for epoca in range(max_epocas):
        inicio_epoca = time.time()  # Início da cronometragem da época
        # Obter a hora atual
        hora_atual = datetime.now()
        # Formatando a hora atual como 'HH:MM:SS'
        hora_formatada = hora_atual.strftime('%H:%M:%S')
        print(f'hora:{hora_formatada}')
        print(f'época:{epoca}')
        modelo.train()
        perda_total, acuracia_total, amostras_total = 0, 0, 0

        for X_lote, y_lote in carregador_treino:
            #print(f'X_lote:{len(X_lote)}')
            #print(f'y_lote:{len(y_lote)}')
            X_lote, y_lote = X_lote.to(device), y_lote.to(device)  # Movendo os lotes de dados para o dispositivo
            otimizador.zero_grad()
            saida = modelo(X_lote)
            perda = criterio(saida, torch.max(y_lote, 1)[1])
            perda.backward()
            otimizador.step()

            perda_total += perda.item() * X_lote.size(0)
            previsoes = torch.max(saida, 1)[1]
            acuracia_total += accuracy_score(torch.max(y_lote, 1)[1].cpu(), previsoes.cpu()) * X_lote.size(0)
            amostras_total += X_lote.size(0)
            #print('for 1 realizado')
        print(f'treinamento da época:{epoca} finalizado')
        perda_media = perda_total / amostras_total
        acuracia_media = acuracia_total / amostras_total

        fim_epoca = time.time()  # Fim da cronometragem da época
        duracao_epoca = fim_epoca - inicio_epoca
        hora_atual = datetime.now()
        # Formatando a hora atual como 'HH:MM:SS'
        hora_formatada = hora_atual.strftime('%H:%M:%S')
        print(f'fim_epoca :{hora_formatada}')
        print(f'duração :{duracao_epoca}')
        # Armazenando dados da época
        dados_treinamento['epoca'].append(epoca)
        dados_treinamento['perda'].append(perda_media)
        dados_treinamento['acuracia'].append(acuracia_media)
        dados_treinamento['duracao_epoca'].append(duracao_epoca)

        # Critérios de parada
        if criterio_parada == 'loss' and perda_media < limiar_parada:
            print(f"Parada antecipada na época {epoca} devido à perda")
            break
        elif criterio_parada == 'accuracy' and acuracia_media > limiar_parada:
            print(f"Parada antecipada na época {epoca} devido à acurácia")
            break

    return modelo, dados_treinamento

# Função para calcular a acurácia
def calcular_acuracia(modelo, X, y):
    modelo.eval()
    print('Modelo:')
    print(modelo)
    with torch.no_grad():
        #print('X:')
        #print(X)
        #X = torch.tensor(X, dtype=torch.float32).to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        #print('X_tensor:')
        #print(X_tensor)
        previsoes = modelo(X_tensor)
        print('previsoes:')
        print(previsoes)
        #classes_previstas = torch.argmax(previsoes, axis=1).numpy()
        #classes_previstas = torch.argmax(previsoes, axis=1).cpu().numpy()
        classes_previstas = torch.argmax(previsoes, dim=1).cpu().numpy()
        print('Classes previstas:')
        print(classes_previstas)
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        classes_reais = np.argmax(y, axis=1)
        print('Classes reais:')
        print(classes_reais)
        #classes_reais = classes_reais.numpy()
        acuracia = np.mean(classes_previstas == classes_reais)
        print('acuracia')
        print(acuracia)
    return classes_previstas, acuracia


# Função para testar múltiplos modelosQBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
def testar_varios_modelos(configuracoes):
    resultados = []
    total_configuracoes = len(configuracoes)
    tamanho=0
    for config in configuracoes:
        #print(f'Iniciando configuração {config}')
        print(f'Iniciando execução da configuração {tamanho} de {total_configuracoes}.')
        numero=config['numero']
        nomenclatura=config['nomenclatura']
        modelo = RNA(config['tamanhos_camadas'], config['funcoes_ativacao'])
        modelo_treinado, historico_treinamento = treinar_modelo(modelo, X_treino, y_treino, **config['parametros_treino'])
        previsao, acuracia = calcular_acuracia(modelo_treinado, X_teste, y_teste)
        resultados.append({
            'numero':numero,
            'nomenclatura': nomenclatura,
            'config': config,
            'previsao':previsao,
            'acuracia_final': f"{acuracia * 100:.2f}",
            'historico_treinamento': historico_treinamento
        })
        tamanho += 1
    return resultados
###########

def generate_correct_sequences(values_list):
    sequences_data = []
    for value in values_list:
        for i in range(1, 11):  # Gerando sequências de tamanho 1 a 10
            sequence = [int(str(value) * i)]
            sequence_data = {
                'Número': value,
                'Partições': sequence,
                'Conferencia_Partições': True if all(x == str(value) for x in str(sequence[0])) else False,  # Verifica se todos os números são iguais a 'value'
                'Nomenclatura': str(sequence[0])  # Converte a sequência para string
            }
            sequences_data.append(sequence_data)

    # Convertendo para DataFrame
    sequences_df = pd.DataFrame(sequences_data)
    sequences_df['Conferencia_Nomenclatura'] = sequences_df['Nomenclatura'].apply(len)  # Contagem de caracteres na nomenclatura

    return sequences_df

# Gerando sequências corrigidas para o valor especificado
#sequences_df_corrected = generate_correct_sequences(valores)

# Mostrando as primeiras linhas do DataFrame corrigido para visualizar a correção
#sequences_df_corrected.head(10)

valores=[7]
#valores=[2]
#valores=[3,6]
partitions_df = generate_correct_sequences(valores)

if len(partitions_df)>0:
    print(f'Partições e Nomenclaturas dos números {valores} criadas com sucesso: {partitions_df}')
    #pass
    print(partitions_df)

    len_X0 = len(X[0])
    len_y_onehot0 = len(y_onehot[0])

    configuracoes = []

    # Definindo os parâmetros de treino fixos
    parametros_treino = {
        'taxa_aprendizado': 0.01, 
        'tamanho_lote': 1000, 
        'escolha_otimizador': 'SGD', 
        'momento': 0.9, 
        'criterio_parada': 'loss', 
        'limiar_parada': 0.00001, 
        'max_epocas': 100
    }

    for index, row in partitions_df.iterrows():
        # Criando a lista de tamanhos de camada
        tamanhos_camadas = [len_X0] + row['Partições'] + [len_y_onehot0]
        
        # Adicionando a configuração ao dicionário
        configuracoes.append({
            'nomenclatura': row['Nomenclatura'],
            'numero': row['Número'],
            'tamanhos_camadas': tamanhos_camadas,
            #'funcoes_ativacao': ['torch.relu', 'torch.sigmoid'],  # Assumindo funções de ativação
        })

    for config in configuracoes:
        # O número de funções de ativação deve ser len(tamanhos_camadas) - 2
        # pois não se aplica ativação na camada de input e na camada de output
        num_funcoes_ativacao = len(config['tamanhos_camadas']) - 2
        # Ajustando 'funcoes_ativacao' para ter 'torch.relu' para todas exceto a última que deve ser 'torch.sigmoid'
        config['funcoes_ativacao'] = [torch.relu] * (num_funcoes_ativacao) + [torch.sigmoid]
        config['parametros_treino'] = parametros_treino

    #configuracoes

    # # Configurações de exemplo para testar
    # configuracoes = [
    #     {'tamanhos_camadas': [len(X[0]), 6, len(y_onehot[0])], 'funcoes_ativacao': [torch.relu, torch.sigmoid], 'parametros_treino': {'taxa_aprendizado': 0.01, 'tamanho_lote': 10000, 'escolha_otimizador': 'SGD', 'momento': 0.9, 'criterio_parada': 'loss', 'limiar_parada': 0.00001, 'max_epocas': 10}},
    # ]

    print('Iniciando treinamento')
    # Testar os modelos
    resultados_modelos = testar_varios_modelos(configuracoes)
    print('Treinamento finalizado')
    print('Salvando arquivo do treinamento')
    # Converter os resultados em um DataFrame
    df_resultados = pd.DataFrame(resultados_modelos)
    df_resultados['config']=df_resultados.index
    for indice, resultado in df_resultados.iterrows():
        plt.figure()  # Cria uma nova figura para cada gráfico
        plt.plot(resultado['historico_treinamento']['epoca'], resultado['historico_treinamento']['acuracia'], label=f"Config {indice}")
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.title(f'EXTRA 7 - Acurácia do Modelo ao Longo das Épocas - Config {indice}')
        plt.legend()
        # Define o caminho completo para salvar a figura
        caminho_salvamento = f'/home/ibpad/mestrado_raquel/plots_7/extra_acuracia_config_{indice}.png'
        plt.savefig(caminho_salvamento)
        plt.close()
    # # Visualização dos resultados
    # for indice, resultado in df_resultados.iterrows():
    #     plt.plot(resultado['historico_treinamento']['epoca'], resultado['historico_treinamento']['acuracia'], label=f"Config {indice}")
    #     plt.xlabel('Época')
    #     plt.ylabel('Acurácia')
    #     plt.title('Acurácia do Modelo ao Longo das Épocas')
    #     plt.legend()
    #     #plt.show()
    #     caminho_salvamento = f'/home/ibpad/mestrado_raquel/plots/acuracia_config_{indice}.png'
    #     plt.savefig(caminho_salvamento)
    #     plt.close()  # Fecha a figura para evitar o uso excessivo de memória
    print(df_resultados)

    df_resultados.to_csv('/home/ibpad/mestrado_raquel/resultados_teste_oficial_7_extra.csv',index=False)

    print('Arqivo salvo.')

