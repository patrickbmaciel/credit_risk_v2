# 0) Configurações iniciais

# Importando pacotes
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Definindo o estilo visual comum às figuras de avaliação
sns.set_theme(style="whitegrid")

# Definindo os caminhos relativos usados no script
arquivo_processado = "data/processed/credit_risk_processed.csv"
diretorio_tabelas = "outputs/tables"
diretorio_figuras = "outputs/figures"

# Definindo a semente, a proporção de teste e o limite de classificação
semente = 123
proporcao_teste = 0.30
limiar = 0.5

# Listando as 12 colunas esperadas na base processada
colunas_tratadas = [
    "status_emprestimo",
    "idade",
    "renda_anual",
    "residencia",
    "tempo_emprego",
    "intencao_emprestimo",
    "classificacao_emprestimo",
    "valor_emprestimo",
    "taxa_juros_emprestimo",
    "percentual_renda_emprestimo",
    "historico_inadimplencia",
    "duracao_historico_credito",
]

# Identificando a variável que os modelos devem prever
variavel_alvo = "status_emprestimo"

# Listando as variáveis categóricas que precisam ser codificadas
variaveis_categoricas = [
    "residencia",
    "intencao_emprestimo",
    "classificacao_emprestimo",
]

# Listando as variáveis numéricas usadas diretamente pelos modelos
variaveis_numericas = [
    coluna
    for coluna in colunas_tratadas
    if coluna not in variaveis_categoricas + [variavel_alvo]
]

# Definindo função que calcula métricas, lucro, matriz de confusão e pontos
# das curvas ROC e precisão-recall, onde:
# modelo: modelo já treinado;
# x_teste_modelo: variáveis explicativas do teste, preparadas conforme o modelo;
# x_teste_original: variáveis explicativas do teste antes da preparação;
# y_teste: resultados reais do teste;
# nome: nome do algoritmo, como "Regressão Logística";
# estrategia: versão avaliada, como "Original" ou "Balanceada".
def calcular_resultados(
    modelo,
    x_teste_modelo,
    x_teste_original,
    y_teste,
    nome,
    estrategia,
):
    # Localizando a coluna de probabilidades da classe positiva igual a um
    indice_classe_positiva = int(np.flatnonzero(modelo.classes_ == 1)[0])
    # Calculando a probabilidade de inadimplência para cada observação de teste
    probabilidades = modelo.predict_proba(x_teste_modelo)[
        :, indice_classe_positiva
    ]
    # Convertendo probabilidades em previsões usando o limite de 0,5
    previsoes = (probabilidades >= limiar).astype(int)
    # Calculando a matriz de confusão na ordem das classes zero e um
    matriz = confusion_matrix(y_teste, previsoes, labels=[0, 1])
    # Separando os quatro tipos de resultado da matriz de confusão
    verdadeiro_negativo, falso_positivo, falso_negativo, verdadeiro_positivo = (
        matriz.ravel()
    )
    # Separando os valores e as taxas dos empréstimos avaliados
    valores_emprestimos = x_teste_original["valor_emprestimo"].to_numpy()
    taxas_juros = (
        x_teste_original["taxa_juros_emprestimo"].to_numpy() / 100
    )
    resultados_reais = y_teste.to_numpy()
    # Calculando o resultado financeiro dos empréstimos que seriam aprovados
    lucros_emprestimos = np.where(
        (previsoes == 0) & (resultados_reais == 0),
        valores_emprestimos * taxas_juros,
        np.where(
            (previsoes == 0) & (resultados_reais == 1),
            -valores_emprestimos,
            0,
        ),
    )
    # Somando os juros recebidos e as perdas com inadimplência
    lucro = float(lucros_emprestimos.sum())
    # Calculando os pontos usados para desenhar a curva ROC
    falso_positivo_roc, verdadeiro_positivo_roc, _ = roc_curve(
        y_teste,
        probabilidades,
        pos_label=1,
    )
    # Calculando os pontos usados para desenhar a curva precisão-recall
    precisao_pr, recall_pr, _ = precision_recall_curve(
        y_teste,
        probabilidades,
        pos_label=1,
    )
    # Reunindo as contagens, o lucro e as métricas do modelo em um dicionário
    metricas = {
        "modelo": nome,
        "estrategia": estrategia,
        "verdadeiro_negativo": int(verdadeiro_negativo),
        "falso_positivo": int(falso_positivo),
        "falso_negativo": int(falso_negativo),
        "verdadeiro_positivo": int(verdadeiro_positivo),
        "lucro": lucro,
        "acuracia": accuracy_score(y_teste, previsoes),
        "precisao": precision_score(
            y_teste,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            y_teste,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_teste,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_teste, probabilidades),
        "pr_auc": average_precision_score(
            y_teste,
            probabilidades,
            pos_label=1,
        ),
    }
    # Reunindo as coordenadas das curvas ROC e precisão-recall
    curva = (
        falso_positivo_roc,
        verdadeiro_positivo_roc,
        recall_pr,
        precisao_pr,
    )
    # Devolvendo todos os resultados necessários para tabelas e figuras
    return metricas, matriz, curva

# 1) Leitura e validação da base processada

# Lendo a base tratada
base = pd.read_csv(arquivo_processado, encoding="utf-8-sig")

# 2) Divisão entre treino e teste

# Separando as variáveis explicativas da variável que será prevista
x = base.drop(columns=variavel_alvo)
y = base[variavel_alvo].astype(int)

# Criando uma única divisão estratificada com 70% para treino e 30% para teste
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=proporcao_teste,
    random_state=semente,
    stratify=y,
)

# Informando o tamanho e a distribuição dos conjuntos criados
print(f"Registros de treino: {len(y_treino)}")
print(f"Registros de teste: {len(y_teste)}")
print(f"Distribuição do teste: {y_teste.value_counts().sort_index().to_dict()}")
print(f"Proporção de 1 no teste: {(y_teste == 1).mean():.2%}")

# 3) Preparação geral das variáveis

# Criando um objeto que aplicará tratamentos diferentes às variáveis
# numéricas e categóricas
pre_processador_geral = ColumnTransformer(
    # Informando a lista de transformações que serão aplicadas
    transformers=[
        # Definindo o tratamento das variáveis numéricas
        (
            # Nomeando essa etapa como "numericas"
            "numericas",
            # Mantendo as variáveis numéricas sem transformação
            "passthrough",
            # Indicando quais colunas pertencem ao grupo numérico
            variaveis_numericas,
        ),
        # Definindo o tratamento das variáveis categóricas
        (
            # Nomeando essa etapa como "categoricas"
            "categoricas",
            # Criando o codificador que transformará categorias em colunas
            # numéricas
            OneHotEncoder(
                # Removendo a primeira categoria de cada variável
                # e utilizando-a como categoria de referência
                drop="first",
                # Ignorando categorias que apareçam em novos dados, mas 
                # que não tenham sido observadas no treino
                handle_unknown="ignore",
                # Devolvendo uma matriz densa, em vez de uma matriz esparsa
                sparse_output=False,
            ),
            # Indicando quais colunas devem receber o One-Hot Encoding
            variaveis_categoricas,
        ),
    ],
    # Evitando que os nomes das etapas sejam acrescentados
    # aos nomes das variáveis produzidas pelo pré-processamento
    verbose_feature_names_out=False,
)

# Aprendendo as categorias somente com os dados de treino
x_treino_geral = pre_processador_geral.fit_transform(x_treino)

# Aplicando ao teste as categorias aprendidas no treino
x_teste_geral = pre_processador_geral.transform(x_teste)

# Guardando os nomes das variáveis após a codificação categórica
nomes_variaveis_geral = pre_processador_geral.get_feature_names_out()

# 4) Preparação específica para o KNN

# Criando um objeto que aplicará tratamentos diferentes às variáveis 
# numéricas e categóricas antes do treinamento, pois o KNN classifica 
# uma observação calculando sua distância em relação às observações do 
# treino e, por isso, a escala das variáveis influencia diretamente o 
# resultado
pre_processador_knn = ColumnTransformer(
    # Informando a lista de transformações que serão aplicadas
    transformers=[
        # Definindo o tratamento das variáveis numéricas
        (
            # Nomeando essa etapa como "numericas"
            "numericas",
            # Criando o normalizador Min-Max, que normalmente transforma
            # cada variável numérica para a escala entre zero e um
            MinMaxScaler(),
            # Indicando quais colunas devem ser normalizadas
            variaveis_numericas,
        ),
        # Definindo o tratamento das variáveis categóricas
        (
            # Nomeando essa etapa como "categoricas"
            "categoricas",
            # Criando o codificador que transformará categorias
            # em colunas numéricas indicadoras
            OneHotEncoder(
                # Removendo a primeira categoria de cada variável
                # e utilizando-a como categoria de referência
                drop="first",
                # Ignorando categorias que apareçam em novos dados,
                # mas que não tenham sido observadas no conjunto de treino
                handle_unknown="ignore",
                # Devolvendo uma matriz densa, em vez de uma matriz esparsa
                sparse_output=False,
            ),
            # Indicando quais colunas devem receber o One-Hot Encoding
            variaveis_categoricas,
        ),
    ],
    # Evitando que os nomes das etapas sejam acrescentados aos nomes das
    # variáveis produzidas pelo pré-processamento
    verbose_feature_names_out=False,
)

# Aprendendo escalas e categorias somente com os dados de treino
x_treino_knn = pre_processador_knn.fit_transform(x_treino)

# Aplicando ao teste as escalas e categorias aprendidas no treino
x_teste_knn = pre_processador_knn.transform(x_teste)

# Calculando pesos balanceados somente a partir do alvo de treino
pesos_balanceados = compute_sample_weight(
    class_weight="balanced",
    y=y_treino,
)

# Criando o sobreamostrador usado apenas pelo KNN balanceado
sobreamostrador = RandomOverSampler(random_state=semente)

# Balanceando somente os dados de treino usados pelo segundo KNN
x_treino_knn_balanceado, y_treino_knn_balanceado = (
    sobreamostrador.fit_resample(x_treino_knn, y_treino)
)

# 5) Regressão Logística

# Criando a Regressão Logística com a distribuição original do treino
modelo_logistico_original = LogisticRegression(
    class_weight=None,
    max_iter=1000,
    solver="liblinear",
    random_state=semente,
)

# Treinando a Regressão Logística original
modelo_logistico_original.fit(x_treino_geral, y_treino)

# Avaliando a Regressão Logística original no conjunto de teste
(
    metricas_logistica_original,
    matriz_logistica_original,
    curva_logistica_original,
) = calcular_resultados(
    modelo_logistico_original,
    x_teste_geral,
    x_teste,
    y_teste,
    "Regressão Logística",
    "original",
)

# Criando a Regressão Logística com pesos balanceados
modelo_logistico_balanceado = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=semente,
)

# Treinando a Regressão Logística balanceada
modelo_logistico_balanceado.fit(x_treino_geral, y_treino)

# Avaliando a Regressão Logística balanceada no mesmo conjunto de teste
(
    metricas_logistica_balanceada,
    matriz_logistica_balanceada,
    curva_logistica_balanceada,
) = calcular_resultados(
    modelo_logistico_balanceado,
    x_teste_geral,
    x_teste,
    y_teste,
    "Regressão Logística",
    "balanceado",
)

# 6) Árvore de Decisão

# Criando a Árvore de Decisão com a distribuição original do treino
modelo_arvore_original = DecisionTreeClassifier(
    class_weight=None,
    random_state=semente,
)

# Treinando a Árvore de Decisão original
modelo_arvore_original.fit(x_treino_geral, y_treino)

# Avaliando a Árvore de Decisão original no conjunto de teste
metricas_arvore_original, matriz_arvore_original, curva_arvore_original = (
    calcular_resultados(
        modelo_arvore_original,
        x_teste_geral,
        x_teste,
        y_teste,
        "Árvore de Decisão",
        "original",
    )
)

# Criando a Árvore de Decisão com pesos balanceados
modelo_arvore_balanceada = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=semente,
)

# Treinando a Árvore de Decisão balanceada
modelo_arvore_balanceada.fit(x_treino_geral, y_treino)

# Avaliando a Árvore de Decisão balanceada no mesmo conjunto de teste
(
    metricas_arvore_balanceada,
    matriz_arvore_balanceada,
    curva_arvore_balanceada,
) = calcular_resultados(
    modelo_arvore_balanceada,
    x_teste_geral,
    x_teste,
    y_teste,
    "Árvore de Decisão",
    "balanceado",
)

# 7) Gaussian Naive Bayes

# Criando o Gaussian Naive Bayes sem pesos
modelo_naive_original = GaussianNB()

# Treinando o Gaussian Naive Bayes original
modelo_naive_original.fit(x_treino_geral, y_treino)

# Avaliando o Gaussian Naive Bayes original no conjunto de teste
metricas_naive_original, matriz_naive_original, curva_naive_original = (
    calcular_resultados(
        modelo_naive_original,
        x_teste_geral,
        x_teste,
        y_teste,
        "Gaussian Naive Bayes",
        "original",
    )
)

# Criando o segundo Gaussian Naive Bayes
modelo_naive_balanceado = GaussianNB()

# Treinando o Gaussian Naive Bayes com pesos calculados no treino
modelo_naive_balanceado.fit(
    x_treino_geral,
    y_treino,
    sample_weight=pesos_balanceados,
)

# Avaliando o Gaussian Naive Bayes balanceado no mesmo conjunto de teste
(
    metricas_naive_balanceada,
    matriz_naive_balanceada,
    curva_naive_balanceada,
) = calcular_resultados(
    modelo_naive_balanceado,
    x_teste_geral,
    x_teste,
    y_teste,
    "Gaussian Naive Bayes",
    "balanceado",
)

# 8) K-Nearest Neighbors

# Criando o KNN com cinco vizinhos e o treino original
modelo_knn_original = KNeighborsClassifier(n_neighbors=5, n_jobs=1)

# Treinando o KNN original com as variáveis normalizadas
modelo_knn_original.fit(x_treino_knn, y_treino)

# Avaliando o KNN original no conjunto de teste normalizado
metricas_knn_original, matriz_knn_original, curva_knn_original = (
    calcular_resultados(
        modelo_knn_original,
        x_teste_knn,
        x_teste,
        y_teste,
        "KNN",
        "original",
    )
)

# Criando o KNN que receberá o treino sobreamostrado
modelo_knn_balanceado = KNeighborsClassifier(n_neighbors=5, n_jobs=1)

# Treinando o KNN balanceado somente com o treino sobreamostrado
modelo_knn_balanceado.fit(
    x_treino_knn_balanceado,
    y_treino_knn_balanceado,
)

# Avaliando o KNN balanceado no mesmo teste original e desbalanceado
metricas_knn_balanceada, matriz_knn_balanceada, curva_knn_balanceada = (
    calcular_resultados(
        modelo_knn_balanceado,
        x_teste_knn,
        x_teste,
        y_teste,
        "KNN",
        "balanceado",
    )
)

# 9) Random Forest

# Criando a Random Forest com 100 árvores e o treino original
modelo_forest_original = RandomForestClassifier(
    n_estimators=100,
    class_weight=None,
    random_state=semente,
    n_jobs=1,
)

# Treinando a Random Forest original
modelo_forest_original.fit(x_treino_geral, y_treino)

# Avaliando a Random Forest original no conjunto de teste
metricas_forest_original, matriz_forest_original, curva_forest_original = (
    calcular_resultados(
        modelo_forest_original,
        x_teste_geral,
        x_teste,
        y_teste,
        "Random Forest",
        "original",
    )
)

# Criando a Random Forest com pesos balanceados
modelo_forest_balanceado = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=semente,
    n_jobs=1,
)

# Treinando a Random Forest balanceada
modelo_forest_balanceado.fit(x_treino_geral, y_treino)

# Avaliando a Random Forest balanceada no mesmo conjunto de teste
(
    metricas_forest_balanceada,
    matriz_forest_balanceada,
    curva_forest_balanceada,
) = calcular_resultados(
    modelo_forest_balanceado,
    x_teste_geral,
    x_teste,
    y_teste,
    "Random Forest",
    "balanceado",
)

# 10) XGBoost

# Criando o XGBoost com 100 árvores e o treino original
modelo_xgboost_original = XGBClassifier(
    n_estimators=100,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=semente,
    n_jobs=1,
)

# Treinando o XGBoost original
modelo_xgboost_original.fit(x_treino_geral, y_treino)

# Avaliando o XGBoost original no conjunto de teste
metricas_xgboost_original, matriz_xgboost_original, curva_xgboost_original = (
    calcular_resultados(
        modelo_xgboost_original,
        x_teste_geral,
        x_teste,
        y_teste,
        "XGBoost",
        "original",
    )
)

# Criando o XGBoost que receberá os pesos balanceados
modelo_xgboost_balanceado = XGBClassifier(
    n_estimators=100,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=semente,
    n_jobs=1,
)

# Treinando o XGBoost balanceado com os pesos calculados no treino
modelo_xgboost_balanceado.fit(
    x_treino_geral,
    y_treino,
    sample_weight=pesos_balanceados,
)

# Avaliando o XGBoost balanceado no mesmo conjunto de teste
(
    metricas_xgboost_balanceado,
    matriz_xgboost_balanceado,
    curva_xgboost_balanceado,
) = calcular_resultados(
    modelo_xgboost_balanceado,
    x_teste_geral,
    x_teste,
    y_teste,
    "XGBoost",
    "balanceado",
)

# 11) LightGBM

# Criando o LightGBM com 100 árvores e o treino original
modelo_lightgbm_original = LGBMClassifier(
    n_estimators=100,
    objective="binary",
    random_state=semente,
    n_jobs=1,
    verbosity=-1,
)

# Treinando o LightGBM original
modelo_lightgbm_original.fit(x_treino_geral, y_treino)

# Avaliando o LightGBM original no conjunto de teste
(
    metricas_lightgbm_original,
    matriz_lightgbm_original,
    curva_lightgbm_original,
) = calcular_resultados(
    modelo_lightgbm_original,
    x_teste_geral,
    x_teste,
    y_teste,
    "LightGBM",
    "original",
)

# Criando o LightGBM que receberá os pesos balanceados
modelo_lightgbm_balanceado = LGBMClassifier(
    n_estimators=100,
    objective="binary",
    random_state=semente,
    n_jobs=1,
    verbosity=-1,
)

# Treinando o LightGBM balanceado com os pesos calculados no treino
modelo_lightgbm_balanceado.fit(
    x_treino_geral,
    y_treino,
    sample_weight=pesos_balanceados,
)

# Avaliando o LightGBM balanceado no mesmo conjunto de teste
(
    metricas_lightgbm_balanceado,
    matriz_lightgbm_balanceado,
    curva_lightgbm_balanceado,
) = calcular_resultados(
    modelo_lightgbm_balanceado,
    x_teste_geral,
    x_teste,
    y_teste,
    "LightGBM",
    "balanceado",
)

# 12) Tabela de métricas

# Reunindo as métricas na ordem usada para comparar os modelos
linhas_metricas = [
    metricas_logistica_original,
    metricas_logistica_balanceada,
    metricas_arvore_original,
    metricas_arvore_balanceada,
    metricas_naive_original,
    metricas_naive_balanceada,
    metricas_knn_original,
    metricas_knn_balanceada,
    metricas_forest_original,
    metricas_forest_balanceada,
    metricas_xgboost_original,
    metricas_xgboost_balanceado,
    metricas_lightgbm_original,
    metricas_lightgbm_balanceado,
]

# Transformando os resultados em uma tabela com 14 linhas
tabela_metricas = pd.DataFrame(linhas_metricas)

# Localizando o modelo champion pelo maior lucro e, em empate, pelo menor
# número de falsos negativos
indice_champion = tabela_metricas.sort_values(
    ["lucro", "falso_negativo"],
    ascending=[False, True],
).index[0]

# Retirando o champion antes de selecionar um challenger distinto
candidatos_challenger = tabela_metricas.drop(index=indice_champion)

# Localizando o challenger pelo menor número de falsos negativos e, em
# empate, pelo maior lucro
indice_challenger = candidatos_challenger.sort_values(
    ["falso_negativo", "lucro"],
    ascending=[True, False],
).index[0]

# Criando uma coluna que identifica os modelos selecionados
tabela_metricas.insert(2, "papel", "")
tabela_metricas.loc[indice_champion, "papel"] = "champion"
tabela_metricas.loc[indice_challenger, "papel"] = "challenger"

# Listando as métricas que devem ser arredondadas
colunas_decimais = [
    "acuracia",
    "precisao",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]

# Arredondando as métricas para seis casas decimais
tabela_metricas[colunas_decimais] = tabela_metricas[colunas_decimais].round(6)

# Arredondando o lucro para duas casas decimais
tabela_metricas["lucro"] = tabela_metricas["lucro"].round(2)

# Salvando a tabela principal de comparação dos modelos
tabela_metricas.to_csv(
    "outputs/tables/metricas_modelos.csv",
    index=False,
    encoding="utf-8-sig",
)

# Informando as combinações selecionadas como champion e challenger
print(
    "Champion:",
    tabela_metricas.loc[indice_champion, "modelo"],
    tabela_metricas.loc[indice_champion, "estrategia"],
)
print(
    "Challenger:",
    tabela_metricas.loc[indice_challenger, "modelo"],
    tabela_metricas.loc[indice_challenger, "estrategia"],
)

# 13) Coeficientes das Regressões Logísticas

# Criando uma lista vazia para os coeficientes das duas regressões
linhas_coeficientes = []

# Reunindo os modelos logísticos e os nomes de suas estratégias
regressoes_logisticas = [
    ("original", modelo_logistico_original),
    ("balanceado", modelo_logistico_balanceado),
]

# Extraindo o intercepto e os coeficientes de cada regressão
for estrategia, modelo in regressoes_logisticas:
    # Registrando o intercepto estimado pelo modelo
    linhas_coeficientes.append(
        {
            "estrategia": estrategia,
            "variavel": "intercepto",
            "coeficiente": float(modelo.intercept_[0]),
        }
    )
    # Relacionando cada variável ao seu coeficiente estimado
    for variavel, coeficiente in zip(
        nomes_variaveis_geral,
        modelo.coef_[0],
    ):
        # Acrescentando o coeficiente à tabela final
        linhas_coeficientes.append(
            {
                "estrategia": estrategia,
                "variavel": variavel,
                "coeficiente": float(coeficiente),
            }
        )

# Transformando os coeficientes em uma tabela
tabela_coeficientes = pd.DataFrame(linhas_coeficientes)

# Arredondando os coeficientes para oito casas decimais
tabela_coeficientes["coeficiente"] = tabela_coeficientes[
    "coeficiente"
].round(8)

# Salvando os coeficientes das duas regressões
tabela_coeficientes.to_csv(
    "outputs/tables/coeficientes_regressao_logistica.csv",
    index=False,
    encoding="utf-8-sig",
)

# 14) Importâncias dos modelos baseados em árvores

# Nota: A importância nativa das variáveis mede quanto cada variável participou
# das divisões dos modelos baseados em árvores. A forma de cálculo e a escala
# podem variar entre os algoritmos, por isso os valores devem ser comparados
# dentro de cada modelo. Essa medida oferece uma visão global, mas não informa
# se a variável aumenta ou reduz a probabilidade de inadimplência, não
# representa causalidade e não corresponde a métodos como SHAP ou LIME.

# Criando uma lista vazia para as importâncias das variáveis
linhas_importancias = []

# Reunindo os oito modelos que calculam importâncias
modelos_importancias = [
    ("Árvore de Decisão", "original", modelo_arvore_original),
    ("Árvore de Decisão", "balanceado", modelo_arvore_balanceada),
    ("Random Forest", "original", modelo_forest_original),
    ("Random Forest", "balanceado", modelo_forest_balanceado),
    ("XGBoost", "original", modelo_xgboost_original),
    ("XGBoost", "balanceado", modelo_xgboost_balanceado),
    ("LightGBM", "original", modelo_lightgbm_original),
    ("LightGBM", "balanceado", modelo_lightgbm_balanceado),
]

# Extraindo as importâncias calculadas por cada modelo
for nome_modelo, estrategia, modelo in modelos_importancias:
    # Relacionando cada variável à importância calculada
    for variavel, importancia in zip(
        nomes_variaveis_geral,
        modelo.feature_importances_,
    ):
        # Acrescentando a importância à tabela final
        linhas_importancias.append(
            {
                "modelo": nome_modelo,
                "estrategia": estrategia,
                "variavel": variavel,
                "importancia": float(importancia),
            }
        )

# Transformando as importâncias em uma tabela
tabela_importancias = pd.DataFrame(linhas_importancias)

# Arredondando as importâncias para oito casas decimais
tabela_importancias["importancia"] = tabela_importancias[
    "importancia"
].round(8)

# Salvando as importâncias dos modelos baseados em árvores
tabela_importancias.to_csv(
    "outputs/tables/importancias_variaveis.csv",
    index=False,
    encoding="utf-8-sig",
)

# Selecionando as importâncias do modelo champion
importancias_xgboost_balanceado = tabela_importancias.loc[
    (tabela_importancias["modelo"] == "XGBoost")
    & (tabela_importancias["estrategia"] == "balanceado")
]

# Identificando e ordenando as dez variáveis mais importantes
importancias_xgboost_balanceado = (
    importancias_xgboost_balanceado.nlargest(10, "importancia")
    .sort_values("importancia")
)

# Criando o gráfico de barras horizontais do modelo champion
figura_importancias, eixo = plt.subplots(figsize=(10, 7))
eixo.barh(
    importancias_xgboost_balanceado["variavel"].str.replace("_", " ").str.capitalize(),
    importancias_xgboost_balanceado["importancia"],
    color="darkorange",
)

# Ajustando o eixo e o título da figura
eixo.set_xlabel("Importância")
eixo.set_ylabel("")
eixo.set_title(
    "Importâncias das variáveis - XGBoost balanceado",
    fontweight="bold",
)

# Ajustando e salvando a figura
figura_importancias.tight_layout()
figura_importancias.savefig(
    "outputs/figures/importancias_xgboost_balanceado_top10.png",
    dpi=180,
    bbox_inches="tight",
)

# 15) SHAP do modelo champion

# Organizando os dados de teste com nomes legíveis para a figura
x_teste_shap = pd.DataFrame(
    x_teste_geral,
    columns=nomes_variaveis_geral,
)
x_teste_shap.columns = (
    x_teste_shap.columns.str.replace("_", " ").str.capitalize()
)

# Calculando a contribuição de cada variável para as previsões do champion
explicador_shap = shap.TreeExplainer(modelo_xgboost_balanceado)
valores_shap = explicador_shap(x_teste_shap)

# Criando o gráfico que mostra a importância e a direção dos impactos
shap.plots.beeswarm(
    valores_shap,
    max_display=10,
    show=False,
    plot_size=(10, 7),
    color_bar_label="Valor da variável",
    group_remaining_features=False,
)

# Ajustando os textos da figura
figura_shap = plt.gcf()
eixo_shap = plt.gca()
eixo_shap.set_xlabel("Valor SHAP (impacto na previsão)")
eixo_shap.set_title(
    "SHAP do XGBoost balanceado",
    fontweight="bold",
)
figura_shap.axes[-1].set_yticklabels(["Baixo", "Alto"])

# Ajustando e salvando a figura
figura_shap.tight_layout()
figura_shap.savefig(
    "outputs/figures/shap_xgboost_balanceado.png",
    dpi=180,
    bbox_inches="tight",
)

# 16) Matrizes de confusão e curvas ROC

# Reunindo os resultados necessários para as figuras dos sete algoritmos
comparacoes_modelos = [
    (
        "regressao_logistica",
        "Regressão Logística",
        matriz_logistica_original,
        matriz_logistica_balanceada,
        curva_logistica_original,
        curva_logistica_balanceada,
        metricas_logistica_original["roc_auc"],
        metricas_logistica_balanceada["roc_auc"],
        metricas_logistica_original["pr_auc"],
        metricas_logistica_balanceada["pr_auc"],
    ),
    (
        "arvore_decisao",
        "Árvore de Decisão",
        matriz_arvore_original,
        matriz_arvore_balanceada,
        curva_arvore_original,
        curva_arvore_balanceada,
        metricas_arvore_original["roc_auc"],
        metricas_arvore_balanceada["roc_auc"],
        metricas_arvore_original["pr_auc"],
        metricas_arvore_balanceada["pr_auc"],
    ),
    (
        "naive_bayes",
        "Gaussian Naive Bayes",
        matriz_naive_original,
        matriz_naive_balanceada,
        curva_naive_original,
        curva_naive_balanceada,
        metricas_naive_original["roc_auc"],
        metricas_naive_balanceada["roc_auc"],
        metricas_naive_original["pr_auc"],
        metricas_naive_balanceada["pr_auc"],
    ),
    (
        "knn",
        "KNN",
        matriz_knn_original,
        matriz_knn_balanceada,
        curva_knn_original,
        curva_knn_balanceada,
        metricas_knn_original["roc_auc"],
        metricas_knn_balanceada["roc_auc"],
        metricas_knn_original["pr_auc"],
        metricas_knn_balanceada["pr_auc"],
    ),
    (
        "random_forest",
        "Random Forest",
        matriz_forest_original,
        matriz_forest_balanceada,
        curva_forest_original,
        curva_forest_balanceada,
        metricas_forest_original["roc_auc"],
        metricas_forest_balanceada["roc_auc"],
        metricas_forest_original["pr_auc"],
        metricas_forest_balanceada["pr_auc"],
    ),
    (
        "xgboost",
        "XGBoost",
        matriz_xgboost_original,
        matriz_xgboost_balanceado,
        curva_xgboost_original,
        curva_xgboost_balanceado,
        metricas_xgboost_original["roc_auc"],
        metricas_xgboost_balanceado["roc_auc"],
        metricas_xgboost_original["pr_auc"],
        metricas_xgboost_balanceado["pr_auc"],
    ),
    (
        "lightgbm",
        "LightGBM",
        matriz_lightgbm_original,
        matriz_lightgbm_balanceado,
        curva_lightgbm_original,
        curva_lightgbm_balanceado,
        metricas_lightgbm_original["roc_auc"],
        metricas_lightgbm_balanceado["roc_auc"],
        metricas_lightgbm_original["pr_auc"],
        metricas_lightgbm_balanceado["pr_auc"],
    ),
]

# Definindo as cores usadas nas duas curvas ROC de cada algoritmo
cores_roc = {"original": "steelblue", "balanceado": "darkorange"}

# Criando uma figura de matriz e uma figura ROC para cada algoritmo
for (
    nome_arquivo,
    titulo_modelo,
    matriz_original,
    matriz_balanceada,
    curva_original,
    curva_balanceada,
    area_original,
    area_balanceada,
    area_pr_original,
    area_pr_balanceada,
) in comparacoes_modelos:
    # Criando duas áreas lado a lado para as matrizes de confusão
    figura_matrizes, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    # Reunindo as estratégias e matrizes que serão desenhadas
    matrizes_estrategias = [
        ("original", matriz_original),
        ("balanceado", matriz_balanceada),
    ]
    # Desenhando uma matriz de confusão em cada área
    for eixo, (estrategia, matriz) in zip(eixos, matrizes_estrategias):
        # Representando as contagens da matriz por números e cores
        sns.heatmap(
            matriz,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Não inadimplente", "Inadimplente"],
            yticklabels=["Não inadimplente", "Inadimplente"],
            ax=eixo,
        )
        # Identificando a estratégia, a previsão e a realidade
        eixo.set_title(estrategia.capitalize())
        eixo.set_xlabel("Previsão")
        eixo.set_ylabel("Realidade")
    # Inserindo o título geral das matrizes do algoritmo
    figura_matrizes.suptitle(
        f"Matrizes de confusão - {titulo_modelo}",
        fontsize=14,
        fontweight="bold",
    )
    # Ajustando e salvando a figura das matrizes
    figura_matrizes.tight_layout()
    caminho_matriz = f"outputs/figures/matriz_confusao_{nome_arquivo}.png"
    figura_matrizes.savefig(caminho_matriz, dpi=180, bbox_inches="tight")
    # Criando a área de desenho das duas curvas ROC
    figura_roc, eixo = plt.subplots(figsize=(7, 6))
    # Reunindo estratégias, curvas e áreas sob a curva
    curvas_estrategias = [
        ("original", curva_original, area_original),
        ("balanceado", curva_balanceada, area_balanceada),
    ]
    # Desenhando uma curva ROC para cada estratégia
    for estrategia, curva, area in curvas_estrategias:
        # Separando as taxas de falsos e verdadeiros positivos
        falso_positivo_roc, verdadeiro_positivo_roc = curva[:2]
        # Desenhando a curva e informando a área na legenda
        eixo.plot(
            falso_positivo_roc,
            verdadeiro_positivo_roc,
            color=cores_roc[estrategia],
            linewidth=2,
            label=f"{estrategia.capitalize()} (AUC = {area:.3f})",
        )
    # Desenhando a referência de um classificador aleatório
    eixo.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
    # Definindo os limites, rótulos, título e legenda da curva ROC
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.02)
    eixo.set_xlabel("Taxa de falsos positivos")
    eixo.set_ylabel("Taxa de verdadeiros positivos")
    eixo.set_title(f"Curvas ROC - {titulo_modelo}", fontweight="bold")
    eixo.legend(loc="lower right")
    # Ajustando e salvando a figura das curvas ROC
    figura_roc.tight_layout()
    caminho_roc = f"outputs/figures/curva_roc_{nome_arquivo}.png"
    figura_roc.savefig(caminho_roc, dpi=180, bbox_inches="tight")

# 17) Curvas precisão-recall comparativas

# Definindo uma cor para representar cada algoritmo
cores_modelos = {
    "regressao_logistica": "steelblue",
    "arvore_decisao": "firebrick",
    "naive_bayes": "saddlebrown",
    "knn": "mediumpurple",
    "random_forest": "teal",
    "xgboost": "darkorange",
    "lightgbm": "forestgreen",
}

# Calculando a proporção da classe positiva usada como referência
proporcao_positiva = (y_teste == 1).mean()

# Criando dois painéis para separar as estratégias original e balanceada
figura_pr_estrategias, eixos = plt.subplots(1, 2, figsize=(16, 6.5))

# Desenhando as curvas dos sete algoritmos em seus respectivos painéis
for (
    nome_arquivo,
    titulo_modelo,
    matriz_original,
    matriz_balanceada,
    curva_original,
    curva_balanceada,
    area_original,
    area_balanceada,
    area_pr_original,
    area_pr_balanceada,
) in comparacoes_modelos:
    # Separando recall e precisão da estratégia original
    recall_original, precisao_original = curva_original[2:]
    # Desenhando a curva original no primeiro painel
    eixos[0].plot(
        recall_original,
        precisao_original,
        color=cores_modelos[nome_arquivo],
        linewidth=2,
        label=f"{titulo_modelo} (PR-AUC = {area_pr_original:.3f})",
    )
    # Separando recall e precisão da estratégia balanceada
    recall_balanceada, precisao_balanceada = curva_balanceada[2:]
    # Desenhando a curva balanceada no segundo painel
    eixos[1].plot(
        recall_balanceada,
        precisao_balanceada,
        color=cores_modelos[nome_arquivo],
        linewidth=2,
        label=f"{titulo_modelo} (PR-AUC = {area_pr_balanceada:.3f})",
    )

# Ajustando os dois painéis e inserindo a referência da classe positiva
for eixo, titulo in zip(
    eixos,
    ["Estratégias originais", "Estratégias balanceadas"],
):
    eixo.axhline(
        proporcao_positiva,
        color="gray",
        linestyle="--",
        label=f"Prevalência = {proporcao_positiva:.3f}",
    )
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.02)
    eixo.set_xlabel("Recall")
    eixo.set_ylabel("Precisão")
    eixo.set_title(titulo, fontweight="bold")
    eixo.legend(loc="lower left", fontsize=8)

# Inserindo o título geral e salvando a comparação por estratégia
figura_pr_estrategias.suptitle(
    "Curvas precisão-recall por estratégia",
    fontsize=14,
    fontweight="bold",
)
figura_pr_estrategias.tight_layout()
figura_pr_estrategias.savefig(
    "outputs/figures/curvas_pr_por_estrategia.png",
    dpi=180,
    bbox_inches="tight",
)

# Criando um único eixo para destacar champion e challenger
figura_pr_destaques, eixo = plt.subplots(figsize=(9, 7))

# Desenhando em cinza as 12 combinações que não serão destacadas
for (
    nome_arquivo,
    titulo_modelo,
    matriz_original,
    matriz_balanceada,
    curva_original,
    curva_balanceada,
    area_original,
    area_balanceada,
    area_pr_original,
    area_pr_balanceada,
) in comparacoes_modelos:
    # Desenhando todas as estratégias originais em cinza
    recall_original, precisao_original = curva_original[2:]
    eixo.plot(
        recall_original,
        precisao_original,
        color="lightgray",
        linewidth=1.2,
        alpha=0.7,
    )
    # Desenhando em cinza as estratégias balanceadas não destacadas
    if nome_arquivo not in ["xgboost", "lightgbm"]:
        recall_balanceada, precisao_balanceada = curva_balanceada[2:]
        eixo.plot(
            recall_balanceada,
            precisao_balanceada,
            color="lightgray",
            linewidth=1.2,
            alpha=0.7,
        )

# Separando e destacando a curva do XGBoost balanceado
recall_xgboost, precisao_xgboost = curva_xgboost_balanceado[2:]
eixo.plot(
    recall_xgboost,
    precisao_xgboost,
    color=cores_modelos["xgboost"],
    linewidth=3,
    label=(
        "XGBoost balanceado "
        f"(PR-AUC = {metricas_xgboost_balanceado['pr_auc']:.3f})"
    ),
)

# Separando e destacando a curva do LightGBM balanceado
recall_lightgbm, precisao_lightgbm = curva_lightgbm_balanceado[2:]
eixo.plot(
    recall_lightgbm,
    precisao_lightgbm,
    color=cores_modelos["lightgbm"],
    linewidth=3,
    label=(
        "LightGBM balanceado "
        f"(PR-AUC = {metricas_lightgbm_balanceado['pr_auc']:.3f})"
    ),
)

# Inserindo a referência e ajustando o eixo da figura de destaque
eixo.axhline(
    proporcao_positiva,
    color="gray",
    linestyle="--",
    label=f"Prevalência = {proporcao_positiva:.3f}",
)
eixo.set_xlim(0, 1)
eixo.set_ylim(0, 1.02)
eixo.set_xlabel("Recall")
eixo.set_ylabel("Precisão")
eixo.set_title(
    "Curvas precisão-recall - modelos selecionados em destaque",
    fontweight="bold",
)
eixo.legend(loc="lower left")

# Ajustando e salvando a figura de destaque
figura_pr_destaques.tight_layout()
figura_pr_destaques.savefig(
    "outputs/figures/curvas_pr_modelos_destacados.png",
    dpi=180,
    bbox_inches="tight",
)

# 18) Comparação do lucro estimado

# Ordenando os algoritmos pelo maior lucro entre suas duas estratégias
ordem_modelos_lucro = (
    tabela_metricas.groupby("modelo")["lucro"]
    .max()
    .sort_values(ascending=False)
    .index.tolist()
)

# Invertendo a ordem para apresentar o maior lucro na parte superior
ordem_modelos_lucro = ordem_modelos_lucro[::-1]

# Definindo limites comuns com espaço para os rótulos monetários
lucro_minimo = min(0, tabela_metricas["lucro"].min())
lucro_maximo = max(0, tabela_metricas["lucro"].max())
amplitude_lucro = lucro_maximo - lucro_minimo
limite_inferior_lucro = lucro_minimo - amplitude_lucro * 0.08
limite_superior_lucro = lucro_maximo + amplitude_lucro * 0.18

# Criando a figura que compara as estratégias de cada algoritmo
figura_lucro, eixo = plt.subplots(figsize=(12, 7))

for posicao, modelo in enumerate(ordem_modelos_lucro):
    linhas_modelo = tabela_metricas.loc[tabela_metricas["modelo"] == modelo]
    linha_original = linhas_modelo.loc[
        linhas_modelo["estrategia"] == "original"
    ].iloc[0]
    linha_balanceada = linhas_modelo.loc[
        linhas_modelo["estrategia"] == "balanceado"
    ].iloc[0]

    # Ligando os lucros das estratégias original e balanceada
    eixo.plot(
        [linha_original["lucro"], linha_balanceada["lucro"]],
        [posicao, posicao],
        color="lightgray",
        linewidth=2,
        zorder=1,
    )

    # Desenhando os pontos das duas estratégias
    eixo.scatter(
        linha_original["lucro"],
        posicao,
        color="steelblue",
        edgecolor="black",
        s=90,
        zorder=2,
        label="Original" if posicao == 0 else None,
    )
    eixo.scatter(
        linha_balanceada["lucro"],
        posicao,
        color="darkorange",
        edgecolor="black",
        s=90,
        zorder=2,
        label="Balanceado" if posicao == 0 else None,
    )

    # Formatando e apresentando os lucros em milhões
    rotulo_original = (
        f"$ {abs(linha_original['lucro']) / 1_000_000:.2f} mi"
    ).replace(".", ",")
    if linha_original["lucro"] < 0:
        rotulo_original = f"-{rotulo_original}"

    rotulo_balanceado = (
        f"$ {abs(linha_balanceada['lucro']) / 1_000_000:.2f} mi"
    ).replace(".", ",")
    if linha_balanceada["lucro"] < 0:
        rotulo_balanceado = f"-{rotulo_balanceado}"

    eixo.annotate(
        rotulo_original,
        (linha_original["lucro"], posicao),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
    eixo.annotate(
        rotulo_balanceado,
        (linha_balanceada["lucro"], posicao),
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )

# Criando e formatando as marcações do eixo em milhões
passo_eixo_lucro = 2_000_000
marcacoes_lucro = np.arange(
    np.ceil(limite_inferior_lucro / passo_eixo_lucro) * passo_eixo_lucro,
    limite_superior_lucro,
    passo_eixo_lucro,
)
rotulos_eixo_lucro = [
    f"$ {valor / 1_000_000:.1f} mi".replace(".", ",")
    for valor in marcacoes_lucro
]

# Ajustando os elementos visuais e salvando a figura
eixo.axvline(0, color="black", linewidth=1)
eixo.set_xlim(limite_inferior_lucro, limite_superior_lucro)
eixo.set_yticks(range(len(ordem_modelos_lucro)))
eixo.set_yticklabels(ordem_modelos_lucro)
eixo.set_xticks(marcacoes_lucro)
eixo.set_xticklabels(rotulos_eixo_lucro)
eixo.set_xlabel("Lucro estimado ($)")
eixo.set_ylabel("")
eixo.set_title("Efeito da estratégia no lucro", fontweight="bold")
eixo.legend(loc="lower right")
figura_lucro.tight_layout()
figura_lucro.savefig(
    "outputs/figures/lucro_pontos_conectados.png",
    dpi=180,
    bbox_inches="tight",
)

# 19) Visualização das Árvores de Decisão

# Criando duas áreas para comparar as versões original e balanceada
figura_arvores, eixos = plt.subplots(1, 2, figsize=(28, 10))

# Desenhando os três primeiros níveis da Árvore de Decisão original
plot_tree(
    modelo_arvore_original,
    max_depth=3,
    feature_names=nomes_variaveis_geral,
    class_names=["Não inadimplente", "Inadimplente"],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=6,
    ax=eixos[0],
)

# Identificando a estratégia da primeira árvore
eixos[0].set_title("Árvore original", fontweight="bold")

# Desenhando os três primeiros níveis da Árvore de Decisão balanceada
plot_tree(
    modelo_arvore_balanceada,
    max_depth=3,
    feature_names=nomes_variaveis_geral,
    class_names=["Não inadimplente", "Inadimplente"],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=6,
    ax=eixos[1],
)

# Identificando a estratégia da segunda árvore
eixos[1].set_title("Árvore balanceado", fontweight="bold")

# Inserindo o título geral da comparação das árvores
figura_arvores.suptitle(
    "Árvores de Decisão - três primeiros níveis",
    fontsize=16,
    fontweight="bold",
)

# Ajustando e salvando a figura das árvores
figura_arvores.tight_layout()
figura_arvores.savefig(
    "outputs/figures/arvores_decisao.png",
    dpi=180,
    bbox_inches="tight",
)
