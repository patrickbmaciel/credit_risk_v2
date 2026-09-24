# 0) Configurações iniciais

# Importando pacotes
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
from sklearn.model_selection import StratifiedKFold
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
diretorio_tabelas = Path("outputs/tables")
diretorio_figuras = Path("outputs/figures")
diretorio_tabelas.mkdir(parents=True, exist_ok=True)
diretorio_figuras.mkdir(parents=True, exist_ok=True)

# Definindo as sementes, os folds e o limite de classificação
semente = 123
semente_interna = 456
folds_externos = 10
folds_internos = 5
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
variavel_alvo = "status_emprestimo"
variaveis_categoricas = [
    "residencia",
    "intencao_emprestimo",
    "classificacao_emprestimo",
]
variaveis_numericas = [
    coluna
    for coluna in colunas_tratadas
    if coluna not in variaveis_categoricas + [variavel_alvo]
]


# Criando os dois pré-processadores usados em cada fold. A função garante que
# escalas e categorias sejam sempre aprendidas novamente somente com o treino.
def criar_pre_processadores():
    pre_processador_geral = ColumnTransformer(
        transformers=[
            ("numericas", "passthrough", variaveis_numericas),
            (
                "categoricas",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                variaveis_categoricas,
            ),
        ],
        verbose_feature_names_out=False,
    )
    pre_processador_knn = ColumnTransformer(
        transformers=[
            ("numericas", MinMaxScaler(), variaveis_numericas),
            (
                "categoricas",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                variaveis_categoricas,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return pre_processador_geral, pre_processador_knn


# Criando novas instâncias dos 14 candidatos. Cada chamada devolve modelos
# ainda não ajustados, condição necessária para usá-los em diferentes folds.
def criar_candidatos():
    return [
        {
            "id": "logistica_original",
            "arquivo": "regressao_logistica",
            "modelo": "Regressão Logística",
            "estrategia": "original",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": LogisticRegression(
                class_weight=None,
                max_iter=1000,
                solver="liblinear",
                random_state=semente,
            ),
        },
        {
            "id": "logistica_balanceada",
            "arquivo": "regressao_logistica",
            "modelo": "Regressão Logística",
            "estrategia": "balanceado",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=semente,
            ),
        },
        {
            "id": "arvore_original",
            "arquivo": "arvore_decisao",
            "modelo": "Árvore de Decisão",
            "estrategia": "original",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": DecisionTreeClassifier(
                class_weight=None,
                random_state=semente,
            ),
        },
        {
            "id": "arvore_balanceada",
            "arquivo": "arvore_decisao",
            "modelo": "Árvore de Decisão",
            "estrategia": "balanceado",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=semente,
            ),
        },
        {
            "id": "naive_original",
            "arquivo": "naive_bayes",
            "modelo": "Gaussian Naive Bayes",
            "estrategia": "original",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": GaussianNB(),
        },
        {
            "id": "naive_balanceado",
            "arquivo": "naive_bayes",
            "modelo": "Gaussian Naive Bayes",
            "estrategia": "balanceado",
            "dados": "geral",
            "balanceamento": "pesos",
            "estimador": GaussianNB(),
        },
        {
            "id": "knn_original",
            "arquivo": "knn",
            "modelo": "KNN",
            "estrategia": "original",
            "dados": "knn",
            "balanceamento": "nenhum",
            "estimador": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        },
        {
            "id": "knn_balanceado",
            "arquivo": "knn",
            "modelo": "KNN",
            "estrategia": "balanceado",
            "dados": "knn",
            "balanceamento": "sobreamostragem",
            "estimador": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        },
        {
            "id": "forest_original",
            "arquivo": "random_forest",
            "modelo": "Random Forest",
            "estrategia": "original",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": RandomForestClassifier(
                n_estimators=100,
                class_weight=None,
                random_state=semente,
                n_jobs=1,
            ),
        },
        {
            "id": "forest_balanceada",
            "arquivo": "random_forest",
            "modelo": "Random Forest",
            "estrategia": "balanceado",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=semente,
                n_jobs=1,
            ),
        },
        {
            "id": "xgboost_original",
            "arquivo": "xgboost",
            "modelo": "XGBoost",
            "estrategia": "original",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": XGBClassifier(
                n_estimators=100,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=semente,
                n_jobs=1,
            ),
        },
        {
            "id": "xgboost_balanceado",
            "arquivo": "xgboost",
            "modelo": "XGBoost",
            "estrategia": "balanceado",
            "dados": "geral",
            "balanceamento": "pesos",
            "estimador": XGBClassifier(
                n_estimators=100,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=semente,
                n_jobs=1,
            ),
        },
        {
            "id": "lightgbm_original",
            "arquivo": "lightgbm",
            "modelo": "LightGBM",
            "estrategia": "original",
            "dados": "geral",
            "balanceamento": "nenhum",
            "estimador": LGBMClassifier(
                n_estimators=100,
                objective="binary",
                random_state=semente,
                n_jobs=1,
                verbosity=-1,
            ),
        },
        {
            "id": "lightgbm_balanceado",
            "arquivo": "lightgbm",
            "modelo": "LightGBM",
            "estrategia": "balanceado",
            "dados": "geral",
            "balanceamento": "pesos",
            "estimador": LGBMClassifier(
                n_estimators=100,
                objective="binary",
                random_state=semente,
                n_jobs=1,
                verbosity=-1,
            ),
        },
    ]


# Ajustando os 14 modelos em um treino e calculando probabilidades para uma
# amostra de avaliação. Os tratamentos são aprendidos apenas no treino recebido.
def treinar_modelos(x_treino, y_treino, x_avaliacao):
    pre_processador_geral, pre_processador_knn = criar_pre_processadores()
    x_treino_geral = pre_processador_geral.fit_transform(x_treino)
    x_avaliacao_geral = pre_processador_geral.transform(x_avaliacao)
    x_treino_knn = pre_processador_knn.fit_transform(x_treino)
    x_avaliacao_knn = pre_processador_knn.transform(x_avaliacao)

    # Calculando pesos e sobreamostragem somente a partir do treino do fold
    pesos_balanceados = compute_sample_weight(
        class_weight="balanced",
        y=y_treino,
    )
    sobreamostrador = RandomOverSampler(random_state=semente)
    x_treino_knn_balanceado, y_treino_knn_balanceado = (
        sobreamostrador.fit_resample(x_treino_knn, y_treino)
    )

    modelos_ajustados = {}
    probabilidades = {}
    candidatos = criar_candidatos()

    for candidato in candidatos:
        estimador = candidato["estimador"]
        if candidato["dados"] == "knn":
            x_ajuste = x_treino_knn
            x_previsao = x_avaliacao_knn
        else:
            x_ajuste = x_treino_geral
            x_previsao = x_avaliacao_geral

        if candidato["balanceamento"] == "sobreamostragem":
            estimador.fit(x_treino_knn_balanceado, y_treino_knn_balanceado)
        elif candidato["balanceamento"] == "pesos":
            estimador.fit(x_ajuste, y_treino, sample_weight=pesos_balanceados)
        else:
            estimador.fit(x_ajuste, y_treino)

        indice_positivo = int(np.flatnonzero(estimador.classes_ == 1)[0])
        probabilidades[candidato["id"]] = estimador.predict_proba(x_previsao)[
            :, indice_positivo
        ]
        modelos_ajustados[candidato["id"]] = estimador

    dados_preparados = {
        "geral": x_avaliacao_geral,
        "knn": x_avaliacao_knn,
    }
    return (
        candidatos,
        modelos_ajustados,
        probabilidades,
        pre_processador_geral,
        dados_preparados,
    )


# Calculando métricas, lucro, matriz de confusão e pontos das curvas a partir
# das probabilidades previstas para uma amostra que não foi usada no ajuste.
def calcular_resultados(probabilidades, x_original, y_real, candidato):
    previsoes = (probabilidades >= limiar).astype(int)
    matriz = confusion_matrix(y_real, previsoes, labels=[0, 1])
    verdadeiro_negativo, falso_positivo, falso_negativo, verdadeiro_positivo = (
        matriz.ravel()
    )
    valores = x_original["valor_emprestimo"].to_numpy()
    taxas = x_original["taxa_juros_emprestimo"].to_numpy() / 100
    resultados_reais = y_real.to_numpy()
    lucros_emprestimos = np.where(
        (previsoes == 0) & (resultados_reais == 0),
        valores * taxas,
        np.where(
            (previsoes == 0) & (resultados_reais == 1),
            -valores,
            0,
        ),
    )
    lucro = float(lucros_emprestimos.sum())
    quantidade_positivos = falso_negativo + verdadeiro_positivo
    taxa_falso_negativo = (
        falso_negativo / quantidade_positivos if quantidade_positivos else 0.0
    )
    falso_positivo_roc, verdadeiro_positivo_roc, _ = roc_curve(
        y_real,
        probabilidades,
        pos_label=1,
    )
    precisao_pr, recall_pr, _ = precision_recall_curve(
        y_real,
        probabilidades,
        pos_label=1,
    )
    metricas = {
        "id": candidato["id"],
        "ordem": candidato["ordem"],
        "modelo": candidato["modelo"],
        "estrategia": candidato["estrategia"],
        "verdadeiro_negativo": int(verdadeiro_negativo),
        "falso_positivo": int(falso_positivo),
        "falso_negativo": int(falso_negativo),
        "verdadeiro_positivo": int(verdadeiro_positivo),
        "taxa_falso_negativo": taxa_falso_negativo,
        "lucro": lucro,
        "lucro_por_registro": lucro / len(y_real),
        "acuracia": accuracy_score(y_real, previsoes),
        "precisao": precision_score(
            y_real,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            y_real,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_real,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_real, probabilidades),
        "pr_auc": average_precision_score(
            y_real,
            probabilidades,
            pos_label=1,
        ),
    }
    curva = (
        falso_positivo_roc,
        verdadeiro_positivo_roc,
        recall_pr,
        precisao_pr,
    )
    return metricas, matriz, curva, previsoes


# Resumindo os cinco folds internos e calculando os erros-padrão usados na
# seleção. Os desvios também são mantidos para priorizar maior estabilidade.
def resumir_validacao_interna(tabela_folds):
    resumo = (
        tabela_folds.groupby(
            ["id", "ordem", "modelo", "estrategia"],
            sort=False,
        )
        .agg(
            lucro_por_registro_media=("lucro_por_registro", "mean"),
            lucro_por_registro_desvio=("lucro_por_registro", "std"),
            taxa_falso_negativo_media=("taxa_falso_negativo", "mean"),
            taxa_falso_negativo_desvio=("taxa_falso_negativo", "std"),
            recall_media=("recall", "mean"),
            roc_auc_media=("roc_auc", "mean"),
            pr_auc_media=("pr_auc", "mean"),
            quantidade_folds=("id", "size"),
        )
        .reset_index()
    )
    resumo["lucro_por_registro_erro_padrao"] = (
        resumo["lucro_por_registro_desvio"]
        / np.sqrt(resumo["quantidade_folds"])
    )
    resumo["taxa_falso_negativo_erro_padrao"] = (
        resumo["taxa_falso_negativo_desvio"]
        / np.sqrt(resumo["quantidade_folds"])
    )
    return resumo


# Selecionando champion e challenger pela regra de um erro-padrão. Entre os
# candidatos próximos do melhor resultado, vence aquele com menor dispersão.
def selecionar_champion_challenger(resumo):
    melhor_lucro = resumo.sort_values(
        ["lucro_por_registro_media", "ordem"],
        ascending=[False, True],
    ).iloc[0]
    limite_lucro = (
        melhor_lucro["lucro_por_registro_media"]
        - melhor_lucro["lucro_por_registro_erro_padrao"]
    )
    elegiveis_champion = resumo.loc[
        resumo["lucro_por_registro_media"] >= limite_lucro
    ]
    champion = elegiveis_champion.sort_values(
        [
            "lucro_por_registro_desvio",
            "taxa_falso_negativo_media",
            "lucro_por_registro_media",
            "ordem",
        ],
        ascending=[True, True, False, True],
    ).iloc[0]

    candidatos_challenger = resumo.loc[resumo["id"] != champion["id"]]
    melhor_taxa_fn = candidatos_challenger.sort_values(
        ["taxa_falso_negativo_media", "ordem"],
        ascending=[True, True],
    ).iloc[0]
    limite_taxa_fn = (
        melhor_taxa_fn["taxa_falso_negativo_media"]
        + melhor_taxa_fn["taxa_falso_negativo_erro_padrao"]
    )
    elegiveis_challenger = candidatos_challenger.loc[
        candidatos_challenger["taxa_falso_negativo_media"] <= limite_taxa_fn
    ]
    challenger = elegiveis_challenger.sort_values(
        [
            "taxa_falso_negativo_desvio",
            "lucro_por_registro_media",
            "ordem",
        ],
        ascending=[True, False, True],
    ).iloc[0]
    return champion, challenger


# Executando os cinco folds internos para uma amostra de desenvolvimento.
def executar_validacao_interna(x_desenvolvimento, y_desenvolvimento):
    cv_interno = StratifiedKFold(
        n_splits=folds_internos,
        shuffle=True,
        random_state=semente_interna,
    )
    linhas_folds = []
    for fold_interno, (indices_treino, indices_validacao) in enumerate(
        cv_interno.split(x_desenvolvimento, y_desenvolvimento),
        start=1,
    ):
        x_treino = x_desenvolvimento.iloc[indices_treino]
        y_treino = y_desenvolvimento.iloc[indices_treino]
        x_validacao = x_desenvolvimento.iloc[indices_validacao]
        y_validacao = y_desenvolvimento.iloc[indices_validacao]
        candidatos, _, probabilidades, _, _ = treinar_modelos(
            x_treino,
            y_treino,
            x_validacao,
        )
        for ordem, candidato in enumerate(candidatos):
            candidato["ordem"] = ordem
            metricas, _, _, _ = calcular_resultados(
                probabilidades[candidato["id"]],
                x_validacao,
                y_validacao,
                candidato,
            )
            metricas["fold_interno"] = fold_interno
            linhas_folds.append(metricas)

    tabela_folds = pd.DataFrame(linhas_folds)
    resumo = resumir_validacao_interna(tabela_folds)
    champion, challenger = selecionar_champion_challenger(resumo)
    return tabela_folds, resumo, champion, challenger


# 1) Leitura e validação da base processada

base = pd.read_csv(arquivo_processado, encoding="utf-8-sig")
if list(base.columns) != colunas_tratadas:
    raise ValueError("A base processada não contém as colunas esperadas.")
x = base.drop(columns=variavel_alvo)
y = base[variavel_alvo].astype(int)

# Criando a ordem fixa usada nos desempates
candidatos_referencia = criar_candidatos()
for ordem, candidato in enumerate(candidatos_referencia):
    candidato["ordem"] = ordem

print(f"Registros da base: {len(y)}")
print(f"Proporção de inadimplentes: {(y == 1).mean():.2%}")

# 2) Nested cross-validation estratificado 10x5

cv_externo = StratifiedKFold(
    n_splits=folds_externos,
    shuffle=True,
    random_state=semente,
)
probabilidades_oof = {
    candidato["id"]: np.full(len(base), np.nan)
    for candidato in candidatos_referencia
}
contagem_teste_externo = np.zeros(len(base), dtype=int)
linhas_metricas_folds = []
linhas_selecao_folds = []

for fold_externo, (indices_treino, indices_teste) in enumerate(
    cv_externo.split(x, y),
    start=1,
):
    print(f"Executando fold externo {fold_externo}/{folds_externos}...")
    if np.intersect1d(indices_treino, indices_teste).size:
        raise RuntimeError("Há sobreposição entre treino e teste externos.")

    x_treino_externo = x.iloc[indices_treino]
    y_treino_externo = y.iloc[indices_treino]
    x_teste_externo = x.iloc[indices_teste]
    y_teste_externo = y.iloc[indices_teste]

    # Selecionando os dois papéis somente a partir do treino externo
    _, _, champion_fold, challenger_fold = executar_validacao_interna(
        x_treino_externo,
        y_treino_externo,
    )

    # Ajustando todos os candidatos para gerar comparações externas completas
    candidatos, _, probabilidades, _, _ = treinar_modelos(
        x_treino_externo,
        y_treino_externo,
        x_teste_externo,
    )
    contagem_teste_externo[indices_teste] += 1
    metricas_fold_por_id = {}

    for ordem, candidato in enumerate(candidatos):
        candidato["ordem"] = ordem
        probabilidades_candidato = probabilidades[candidato["id"]]
        probabilidades_oof[candidato["id"]][indices_teste] = (
            probabilidades_candidato
        )
        metricas, _, _, _ = calcular_resultados(
            probabilidades_candidato,
            x_teste_externo,
            y_teste_externo,
            candidato,
        )
        metricas["fold_externo"] = fold_externo
        metricas["registros_teste"] = len(indices_teste)
        metricas["proporcao_inadimplentes_teste"] = (
            y_teste_externo == 1
        ).mean()
        linhas_metricas_folds.append(metricas)
        metricas_fold_por_id[candidato["id"]] = metricas

    # Registrando a escolha interna e o respectivo resultado externo
    for papel, selecionado in [
        ("champion", champion_fold),
        ("challenger", challenger_fold),
    ]:
        metricas_externas = metricas_fold_por_id[selecionado["id"]]
        linhas_selecao_folds.append(
            {
                "fold_externo": fold_externo,
                "papel": papel,
                "id": selecionado["id"],
                "modelo": selecionado["modelo"],
                "estrategia": selecionado["estrategia"],
                "lucro_por_registro_interno_media": selecionado[
                    "lucro_por_registro_media"
                ],
                "lucro_por_registro_interno_erro_padrao": selecionado[
                    "lucro_por_registro_erro_padrao"
                ],
                "taxa_fn_interna_media": selecionado[
                    "taxa_falso_negativo_media"
                ],
                "taxa_fn_interna_erro_padrao": selecionado[
                    "taxa_falso_negativo_erro_padrao"
                ],
                "lucro_externo": metricas_externas["lucro"],
                "lucro_por_registro_externo": metricas_externas[
                    "lucro_por_registro"
                ],
                "falso_negativo_externo": metricas_externas[
                    "falso_negativo"
                ],
                "taxa_fn_externa": metricas_externas[
                    "taxa_falso_negativo"
                ],
                "recall_externo": metricas_externas["recall"],
                "roc_auc_externo": metricas_externas["roc_auc"],
                "pr_auc_externo": metricas_externas["pr_auc"],
            }
        )

# Verificando a cobertura e a validade das previsões externas
if not np.all(contagem_teste_externo == 1):
    raise RuntimeError("Cada registro deve aparecer uma vez no teste externo.")
for identificador, probabilidades in probabilidades_oof.items():
    if np.isnan(probabilidades).any():
        raise RuntimeError(f"Há previsões OOF ausentes para {identificador}.")
    if ((probabilidades < 0) | (probabilidades > 1)).any():
        raise RuntimeError(f"Há probabilidades inválidas para {identificador}.")

# 3) Seleção final e ajuste em toda a base

print("Executando a validação interna final na base completa...")
_, resumo_final, champion_final, challenger_final = executar_validacao_interna(x, y)
print("Champion final:", champion_final["modelo"], champion_final["estrategia"])
print(
    "Challenger final:",
    challenger_final["modelo"],
    challenger_final["estrategia"],
)

# Reajustando os candidatos em todos os registros para interpretação
(
    candidatos_finais,
    modelos_finais,
    _,
    pre_processador_final,
    dados_finais,
) = treinar_modelos(x, y, x)
nomes_variaveis_geral = pre_processador_final.get_feature_names_out()

# 4) Tabelas de métricas e seleção

tabela_metricas_folds = pd.DataFrame(linhas_metricas_folds)
tabela_selecao_folds = pd.DataFrame(linhas_selecao_folds)
resumo_folds_externos = (
    tabela_metricas_folds.groupby(
        ["id", "ordem", "modelo", "estrategia"],
        sort=False,
    )
    .agg(
        lucro_por_registro_media=("lucro_por_registro", "mean"),
        lucro_por_registro_desvio=("lucro_por_registro", "std"),
        taxa_falso_negativo_media=("taxa_falso_negativo", "mean"),
        taxa_falso_negativo_desvio=("taxa_falso_negativo", "std"),
        recall_media=("recall", "mean"),
        recall_desvio=("recall", "std"),
        roc_auc_media=("roc_auc", "mean"),
        roc_auc_desvio=("roc_auc", "std"),
        pr_auc_media=("pr_auc", "mean"),
        pr_auc_desvio=("pr_auc", "std"),
    )
    .reset_index()
)

# Calculando métricas agregadas com todas as previsões OOF
linhas_metricas_modelos = []
resultados_oof = {}
for candidato in candidatos_referencia:
    metricas, matriz, curva, previsoes = calcular_resultados(
        probabilidades_oof[candidato["id"]],
        x,
        y,
        candidato,
    )
    dispersao = resumo_folds_externos.loc[
        resumo_folds_externos["id"] == candidato["id"]
    ].iloc[0]
    for coluna in [
        "lucro_por_registro_media",
        "lucro_por_registro_desvio",
        "taxa_falso_negativo_media",
        "taxa_falso_negativo_desvio",
        "recall_media",
        "recall_desvio",
        "roc_auc_media",
        "roc_auc_desvio",
        "pr_auc_media",
        "pr_auc_desvio",
    ]:
        metricas[coluna] = dispersao[coluna]
    metricas["papel"] = ""
    if candidato["id"] == champion_final["id"]:
        metricas["papel"] = "champion"
    elif candidato["id"] == challenger_final["id"]:
        metricas["papel"] = "challenger"
    linhas_metricas_modelos.append(metricas)
    resultados_oof[candidato["id"]] = {
        "metricas": metricas,
        "matriz": matriz,
        "curva": curva,
        "previsoes": previsoes,
    }

tabela_metricas = pd.DataFrame(linhas_metricas_modelos).sort_values("ordem")

# Contando quantas vezes cada candidato foi selecionado em cada papel
contagem_selecao = (
    tabela_selecao_folds.groupby(["id", "modelo", "estrategia", "papel"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
for papel in ["champion", "challenger"]:
    if papel not in contagem_selecao.columns:
        contagem_selecao[papel] = 0
contagem_selecao = contagem_selecao.rename(
    columns={
        "champion": "vezes_champion",
        "challenger": "vezes_challenger",
    }
)

# Incluindo também os candidatos que não foram selecionados em nenhum fold
tabela_frequencia = pd.DataFrame(
    [
        {
            "id": candidato["id"],
            "modelo": candidato["modelo"],
            "estrategia": candidato["estrategia"],
        }
        for candidato in candidatos_referencia
    ]
).merge(
    contagem_selecao,
    on=["id", "modelo", "estrategia"],
    how="left",
)
tabela_frequencia[["vezes_champion", "vezes_challenger"]] = (
    tabela_frequencia[["vezes_champion", "vezes_challenger"]]
    .fillna(0)
    .astype(int)
)
tabela_frequencia = tabela_frequencia[
    [
        "id",
        "modelo",
        "estrategia",
        "vezes_champion",
        "vezes_challenger",
    ]
]
tabela_frequencia["papel_final"] = ""
tabela_frequencia.loc[
    tabela_frequencia["id"] == champion_final["id"],
    "papel_final",
] = "champion"
tabela_frequencia.loc[
    tabela_frequencia["id"] == challenger_final["id"],
    "papel_final",
] = "challenger"

# Arredondando cópias das tabelas antes de salvá-las
colunas_texto_contagem = [
    "id",
    "ordem",
    "modelo",
    "estrategia",
    "papel",
    "verdadeiro_negativo",
    "falso_positivo",
    "falso_negativo",
    "verdadeiro_positivo",
]
colunas_decimais = [
    coluna
    for coluna in tabela_metricas.columns
    if coluna not in colunas_texto_contagem
]
tabela_metricas_salvar = tabela_metricas.copy()
tabela_metricas_salvar[colunas_decimais] = tabela_metricas_salvar[
    colunas_decimais
].round(6)
tabela_metricas_salvar["lucro"] = tabela_metricas_salvar["lucro"].round(2)
tabela_metricas_salvar.to_csv(
    diretorio_tabelas / "metricas_modelos_ncv.csv",
    index=False,
    encoding="utf-8-sig",
)
tabela_metricas_folds.round(6).to_csv(
    diretorio_tabelas / "metricas_folds_ncv.csv",
    index=False,
    encoding="utf-8-sig",
)
tabela_selecao_folds.round(6).to_csv(
    diretorio_tabelas / "selecao_folds_ncv.csv",
    index=False,
    encoding="utf-8-sig",
)
tabela_frequencia.to_csv(
    diretorio_tabelas / "frequencia_selecao_ncv.csv",
    index=False,
    encoding="utf-8-sig",
)

# 5) Coeficientes das Regressões Logísticas

linhas_coeficientes = []
for estrategia, identificador in [
    ("original", "logistica_original"),
    ("balanceado", "logistica_balanceada"),
]:
    modelo = modelos_finais[identificador]
    linhas_coeficientes.append(
        {
            "estrategia": estrategia,
            "variavel": "intercepto",
            "coeficiente": float(modelo.intercept_[0]),
        }
    )
    for variavel, coeficiente in zip(nomes_variaveis_geral, modelo.coef_[0]):
        linhas_coeficientes.append(
            {
                "estrategia": estrategia,
                "variavel": variavel,
                "coeficiente": float(coeficiente),
            }
        )

tabela_coeficientes = pd.DataFrame(linhas_coeficientes)
tabela_coeficientes["coeficiente"] = tabela_coeficientes[
    "coeficiente"
].round(8)
tabela_coeficientes.to_csv(
    diretorio_tabelas / "coeficientes_regressao_logistica_ncv.csv",
    index=False,
    encoding="utf-8-sig",
)

# 6) Importâncias dos modelos baseados em árvores

modelos_importancias = [
    ("Árvore de Decisão", "original", "arvore_original"),
    ("Árvore de Decisão", "balanceado", "arvore_balanceada"),
    ("Random Forest", "original", "forest_original"),
    ("Random Forest", "balanceado", "forest_balanceada"),
    ("XGBoost", "original", "xgboost_original"),
    ("XGBoost", "balanceado", "xgboost_balanceado"),
    ("LightGBM", "original", "lightgbm_original"),
    ("LightGBM", "balanceado", "lightgbm_balanceado"),
]
linhas_importancias = []
for nome_modelo, estrategia, identificador in modelos_importancias:
    modelo = modelos_finais[identificador]
    for variavel, importancia in zip(
        nomes_variaveis_geral,
        modelo.feature_importances_,
    ):
        linhas_importancias.append(
            {
                "modelo": nome_modelo,
                "estrategia": estrategia,
                "variavel": variavel,
                "importancia": float(importancia),
            }
        )

tabela_importancias = pd.DataFrame(linhas_importancias)
tabela_importancias["importancia"] = tabela_importancias[
    "importancia"
].round(8)
tabela_importancias.to_csv(
    diretorio_tabelas / "importancias_variaveis_ncv.csv",
    index=False,
    encoding="utf-8-sig",
)

# Destacando as dez principais variáveis do XGBoost balanceado
importancias_xgboost_balanceado = (
    tabela_importancias.loc[
        (tabela_importancias["modelo"] == "XGBoost")
        & (tabela_importancias["estrategia"] == "balanceado")
    ]
    .nlargest(10, "importancia")
    .sort_values("importancia")
)
figura_importancias, eixo = plt.subplots(figsize=(10, 7))
eixo.barh(
    importancias_xgboost_balanceado["variavel"]
    .str.replace("_", " ")
    .str.capitalize(),
    importancias_xgboost_balanceado["importancia"],
    color="darkorange",
)
eixo.set_xlabel("Importância")
eixo.set_ylabel("")
eixo.set_title(
    "Importâncias das variáveis - XGBoost balanceado",
    fontweight="bold",
)
figura_importancias.tight_layout()
figura_importancias.savefig(
    diretorio_figuras / "importancias_xgboost_balanceado_top10_ncv.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(figura_importancias)

# 7) SHAP do XGBoost balanceado reajustado na base completa

x_shap = pd.DataFrame(
    dados_finais["geral"],
    columns=nomes_variaveis_geral,
)
x_shap.columns = x_shap.columns.str.replace("_", " ").str.capitalize()
explicador_shap = shap.TreeExplainer(modelos_finais["xgboost_balanceado"])
valores_shap = explicador_shap(x_shap)
shap.plots.beeswarm(
    valores_shap,
    max_display=10,
    show=False,
    plot_size=(10, 7),
    color_bar_label="Valor da variável",
    group_remaining_features=False,
)
figura_shap = plt.gcf()
eixo_shap = plt.gca()
eixo_shap.set_xlabel("Valor SHAP (impacto na previsão)")
eixo_shap.set_title(
    "SHAP do XGBoost balanceado - ajuste final",
    fontweight="bold",
)
figura_shap.axes[-1].set_yticklabels(["Baixo", "Alto"])
figura_shap.tight_layout()
figura_shap.savefig(
    diretorio_figuras / "shap_xgboost_balanceado_ncv.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(figura_shap)

# 8) Matrizes de confusão e curvas ROC out-of-fold

pares_algoritmos = [
    (
        "regressao_logistica",
        "Regressão Logística",
        "logistica_original",
        "logistica_balanceada",
    ),
    (
        "arvore_decisao",
        "Árvore de Decisão",
        "arvore_original",
        "arvore_balanceada",
    ),
    (
        "naive_bayes",
        "Gaussian Naive Bayes",
        "naive_original",
        "naive_balanceado",
    ),
    ("knn", "KNN", "knn_original", "knn_balanceado"),
    (
        "random_forest",
        "Random Forest",
        "forest_original",
        "forest_balanceada",
    ),
    ("xgboost", "XGBoost", "xgboost_original", "xgboost_balanceado"),
    (
        "lightgbm",
        "LightGBM",
        "lightgbm_original",
        "lightgbm_balanceado",
    ),
]
cores_roc = {"original": "steelblue", "balanceado": "darkorange"}

for nome_arquivo, titulo_modelo, id_original, id_balanceado in pares_algoritmos:
    resultado_original = resultados_oof[id_original]
    resultado_balanceado = resultados_oof[id_balanceado]

    figura_matrizes, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    for eixo, estrategia, resultado in [
        (eixos[0], "original", resultado_original),
        (eixos[1], "balanceado", resultado_balanceado),
    ]:
        sns.heatmap(
            resultado["matriz"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Não inadimplente", "Inadimplente"],
            yticklabels=["Não inadimplente", "Inadimplente"],
            ax=eixo,
        )
        eixo.set_title(estrategia.capitalize())
        eixo.set_xlabel("Previsão OOF")
        eixo.set_ylabel("Realidade")
    figura_matrizes.suptitle(
        f"Matrizes de confusão OOF - {titulo_modelo}",
        fontsize=14,
        fontweight="bold",
    )
    figura_matrizes.tight_layout()
    figura_matrizes.savefig(
        diretorio_figuras / f"matriz_confusao_{nome_arquivo}_ncv.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figura_matrizes)

    figura_roc, eixo = plt.subplots(figsize=(7, 6))
    for estrategia, resultado in [
        ("original", resultado_original),
        ("balanceado", resultado_balanceado),
    ]:
        falso_positivo_roc, verdadeiro_positivo_roc = resultado["curva"][:2]
        area = resultado["metricas"]["roc_auc"]
        eixo.plot(
            falso_positivo_roc,
            verdadeiro_positivo_roc,
            color=cores_roc[estrategia],
            linewidth=2,
            label=f"{estrategia.capitalize()} (AUC = {area:.3f})",
        )
    eixo.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.02)
    eixo.set_xlabel("Taxa de falsos positivos")
    eixo.set_ylabel("Taxa de verdadeiros positivos")
    eixo.set_title(f"Curvas ROC OOF - {titulo_modelo}", fontweight="bold")
    eixo.legend(loc="lower right")
    figura_roc.tight_layout()
    figura_roc.savefig(
        diretorio_figuras / f"curva_roc_{nome_arquivo}_ncv.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figura_roc)

# 9) Curvas precisão-recall comparativas

cores_modelos = {
    "regressao_logistica": "steelblue",
    "arvore_decisao": "firebrick",
    "naive_bayes": "saddlebrown",
    "knn": "mediumpurple",
    "random_forest": "teal",
    "xgboost": "darkorange",
    "lightgbm": "forestgreen",
}
proporcao_positiva = (y == 1).mean()
figura_pr_estrategias, eixos = plt.subplots(1, 2, figsize=(16, 6.5))

for nome_arquivo, titulo_modelo, id_original, id_balanceado in pares_algoritmos:
    for eixo, identificador in zip(eixos, [id_original, id_balanceado]):
        resultado = resultados_oof[identificador]
        recall_pr, precisao_pr = resultado["curva"][2:]
        eixo.plot(
            recall_pr,
            precisao_pr,
            color=cores_modelos[nome_arquivo],
            linewidth=2,
            label=(
                f"{titulo_modelo} "
                f"(PR-AUC = {resultado['metricas']['pr_auc']:.3f})"
            ),
        )

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

figura_pr_estrategias.suptitle(
    "Curvas precisão-recall OOF por estratégia",
    fontsize=14,
    fontweight="bold",
)
figura_pr_estrategias.tight_layout()
figura_pr_estrategias.savefig(
    diretorio_figuras / "curvas_pr_por_estrategia_ncv.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(figura_pr_estrategias)

# Destacando os modelos finais selecionados pela validação interna completa
figura_pr_destaques, eixo = plt.subplots(figsize=(9, 7))
for candidato in candidatos_referencia:
    curva = resultados_oof[candidato["id"]]["curva"]
    eixo.plot(
        curva[2],
        curva[3],
        color="lightgray",
        linewidth=1.2,
        alpha=0.7,
    )

for selecionado, papel, cor in [
    (champion_final, "Champion", "darkorange"),
    (challenger_final, "Challenger", "forestgreen"),
]:
    resultado = resultados_oof[selecionado["id"]]
    curva = resultado["curva"]
    eixo.plot(
        curva[2],
        curva[3],
        color=cor,
        linewidth=3,
        label=(
            f"{papel}: {selecionado['modelo']} "
            f"{selecionado['estrategia']} "
            f"(PR-AUC = {resultado['metricas']['pr_auc']:.3f})"
        ),
    )

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
    "Curvas precisão-recall OOF - modelos finais",
    fontweight="bold",
)
eixo.legend(loc="lower left")
figura_pr_destaques.tight_layout()
figura_pr_destaques.savefig(
    diretorio_figuras / "curvas_pr_modelos_destacados_ncv.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(figura_pr_destaques)

# 10) Comparação do lucro estimado out-of-fold

ordem_modelos_lucro = (
    tabela_metricas.groupby("modelo")["lucro"]
    .max()
    .sort_values(ascending=False)
    .index.tolist()[::-1]
)
lucro_minimo = min(0, tabela_metricas["lucro"].min())
lucro_maximo = max(0, tabela_metricas["lucro"].max())
amplitude_lucro = lucro_maximo - lucro_minimo
limite_inferior_lucro = lucro_minimo - amplitude_lucro * 0.08
limite_superior_lucro = lucro_maximo + amplitude_lucro * 0.18

figura_lucro, eixo = plt.subplots(figsize=(12, 7))
for posicao, modelo in enumerate(ordem_modelos_lucro):
    linhas_modelo = tabela_metricas.loc[tabela_metricas["modelo"] == modelo]
    linha_original = linhas_modelo.loc[
        linhas_modelo["estrategia"] == "original"
    ].iloc[0]
    linha_balanceada = linhas_modelo.loc[
        linhas_modelo["estrategia"] == "balanceado"
    ].iloc[0]
    eixo.plot(
        [linha_original["lucro"], linha_balanceada["lucro"]],
        [posicao, posicao],
        color="lightgray",
        linewidth=2,
        zorder=1,
    )
    for linha, cor, estrategia, deslocamento in [
        (linha_original, "steelblue", "Original", 9),
        (linha_balanceada, "darkorange", "Balanceado", -14),
    ]:
        eixo.scatter(
            linha["lucro"],
            posicao,
            color=cor,
            edgecolor="black",
            s=90,
            zorder=2,
            label=estrategia if posicao == 0 else None,
        )
        rotulo = f"$ {abs(linha['lucro']) / 1_000_000:.2f} mi".replace(
            ".", ","
        )
        if linha["lucro"] < 0:
            rotulo = f"-{rotulo}"
        eixo.annotate(
            rotulo,
            (linha["lucro"], posicao),
            xytext=(0, deslocamento),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

passo_eixo_lucro = 5_000_000
marcacoes_lucro = np.arange(
    np.ceil(limite_inferior_lucro / passo_eixo_lucro) * passo_eixo_lucro,
    limite_superior_lucro,
    passo_eixo_lucro,
)
rotulos_eixo_lucro = [
    f"$ {valor / 1_000_000:.1f} mi".replace(".", ",")
    for valor in marcacoes_lucro
]
eixo.axvline(0, color="black", linewidth=1)
eixo.set_xlim(limite_inferior_lucro, limite_superior_lucro)
eixo.set_yticks(range(len(ordem_modelos_lucro)))
eixo.set_yticklabels(ordem_modelos_lucro)
eixo.set_xticks(marcacoes_lucro)
eixo.set_xticklabels(rotulos_eixo_lucro)
eixo.set_xlabel("Lucro estimado OOF ($)")
eixo.set_ylabel("")
eixo.set_title("Efeito da estratégia no lucro OOF", fontweight="bold")
eixo.legend(loc="lower right")
figura_lucro.tight_layout()
figura_lucro.savefig(
    diretorio_figuras / "lucro_pontos_conectados_ncv.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(figura_lucro)

# 11) Visualização das Árvores de Decisão reajustadas

figura_arvores, eixos = plt.subplots(1, 2, figsize=(24, 10))
for eixo, estrategia, identificador in [
    (eixos[0], "Original", "arvore_original"),
    (eixos[1], "Balanceada", "arvore_balanceada"),
]:
    plot_tree(
        modelos_finais[identificador],
        feature_names=nomes_variaveis_geral,
        class_names=["Não inadimplente", "Inadimplente"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=eixo,
    )
    eixo.set_title(f"Árvore de Decisão {estrategia}", fontweight="bold")

figura_arvores.suptitle(
    "Árvores de Decisão - três primeiros níveis do ajuste final",
    fontsize=16,
    fontweight="bold",
)
figura_arvores.tight_layout()
figura_arvores.savefig(
    diretorio_figuras / "arvores_decisao_ncv.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(figura_arvores)

print("Nested cross-validation concluído.")
print("Tabelas salvas em:", diretorio_tabelas)
print("Figuras salvas em:", diretorio_figuras)
