# 0) Configurações iniciais

# Importando pacotes
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Definindo o estilo visual comum a todas as figuras
sns.set_theme(style="whitegrid")

# Definindo os caminhos relativos usados no script
arquivo_processado = "data/processed/credit_risk_processed.csv"
diretorio_tabelas = "outputs/tables"
diretorio_figuras = "outputs/figures"

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

# Listando as variáveis usadas nas frequências e nos gráficos de barras
variaveis_categoricas = [
    "residencia",
    "intencao_emprestimo",
    "classificacao_emprestimo",
]

# Listando as variáveis usadas na matriz de correlação
variaveis_correlacao = [
    "idade",
    "renda_anual",
    "valor_emprestimo",
    "taxa_juros_emprestimo",
    "percentual_renda_emprestimo",
    "duracao_historico_credito",
]

# 1) Leitura da base tratada

# Lendo a base tratada
base = pd.read_csv(arquivo_processado, encoding="utf-8-sig")

# 2) Frequências das variáveis categóricas

# Criando uma lista vazia para reunir as frequências das três variáveis
tabelas_frequencia = []

# Calculando contagens e percentuais para cada variável categórica
for variavel in variaveis_categoricas:
    # Contando as observações de cada categoria e organizando o resultado
    frequencias = (
        base[variavel]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("categoria")
        .reset_index(name="frequencia")
    )
    # Identificando a variável de origem em cada linha da tabela
    frequencias.insert(0, "variavel", variavel)
    # Calculando o percentual de observações de cada categoria
    frequencias["percentual"] = (
        frequencias["frequencia"] / len(base) * 100
    ).round(2)
    # Acrescentando a tabela da variável à lista de resultados
    tabelas_frequencia.append(frequencias)

# Reunindo as três tabelas de frequência em um único DataFrame
tabela_frequencias = pd.concat(tabelas_frequencia, ignore_index=True)

# Salvando as frequências categóricas em CSV
tabela_frequencias.to_csv(
    "outputs/tables/frequencias_categoricas.csv",
    index=False,
    encoding="utf-8-sig",
)

# 3) Matriz de correlação

# Calculando a correlação entre as variáveis numéricas selecionadas
matriz_correlacao = base[variaveis_correlacao].corr().round(2)

# Salvando os coeficientes de correlação em CSV
matriz_correlacao.to_csv(
    "outputs/tables/matriz_correlacao.csv",
    index=True,
    index_label="variavel",
    encoding="utf-8-sig",
)

# 4) Histogramas das variáveis relacionadas aos empréstimos

# Relacionando cada variável ao título que aparecerá no histograma
variaveis_histograma = [
    ("valor_emprestimo", "Valor do empréstimo"),
    ("taxa_juros_emprestimo", "Taxa de juros"),
    ("percentual_renda_emprestimo", "Percentual da renda"),
    ("duracao_historico_credito", "Duração do histórico de crédito"),
]

# Criando uma figura com quatro áreas de desenho
figura_histogramas, eixos = plt.subplots(2, 2, figsize=(12, 8))

# Construindo um histograma em cada área da figura
for eixo, (variavel, titulo) in zip(eixos.flat, variaveis_histograma):
    # Desenhando a distribuição da variável selecionada
    sns.histplot(
        data=base,
        x=variavel,
        bins=30,
        color="steelblue",
        edgecolor="white",
        ax=eixo,
    )
    # Definindo os títulos e os rótulos do histograma
    eixo.set_title(titulo)
    eixo.set_xlabel("")
    eixo.set_ylabel("Frequência")

# Inserindo o título geral da figura
figura_histogramas.suptitle(
    "Distribuições das variáveis relacionadas aos empréstimos",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

# Ajustando os espaços entre os gráficos
figura_histogramas.tight_layout()

# Salvando a figura dos histogramas
figura_histogramas.savefig(
    "outputs/figures/distribuicoes_emprestimos.png",
    dpi=180,
    bbox_inches="tight",
)

# 5) Gráficos de barras das variáveis categóricas

# Relacionando cada variável categórica ao título de seu gráfico
titulos_categoricos = {
    "residencia": "Tipos de residência",
    "intencao_emprestimo": "Intenção de empréstimo",
    "classificacao_emprestimo": "Classificação de empréstimo",
}

# Criando uma figura com três áreas de desenho
figura_barras, eixos = plt.subplots(1, 3, figsize=(18, 6))

# Construindo um gráfico de barras para cada variável categórica
for eixo, variavel in zip(eixos, variaveis_categoricas):
    # Ordenando as categorias que aparecerão no eixo horizontal
    ordem = sorted(base[variavel].unique())
    # Desenhando a contagem de observações de cada categoria
    sns.countplot(
        data=base,
        x=variavel,
        order=ordem,
        color="steelblue",
        ax=eixo,
    )
    # Definindo os títulos, rótulos e rotação dos textos
    eixo.set_title(titulos_categoricos[variavel])
    eixo.set_xlabel("")
    eixo.set_ylabel("Frequência")
    eixo.tick_params(axis="x", rotation=45)

# Inserindo o título geral da figura
figura_barras.suptitle(
    "Distribuições das variáveis categóricas",
    fontsize=15,
    fontweight="bold",
    y=1.03,
)

# Ajustando os espaços entre os gráficos
figura_barras.tight_layout()

# Salvando a figura dos gráficos de barras
figura_barras.savefig(
    "outputs/figures/distribuicoes_categoricas.png",
    dpi=180,
    bbox_inches="tight",
)

# 6) Mapa de calor da matriz de correlação

# Criando a área de desenho do mapa de calor
figura_correlacao, eixo = plt.subplots(figsize=(10, 7))

# Representando os coeficientes por números e cores
sns.heatmap(
    matriz_correlacao,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    ax=eixo,
)

# Inserindo o título do mapa de calor
eixo.set_title("Matriz de correlação entre variáveis", fontweight="bold")

# Ajustando os espaços entre os gráficos
figura_correlacao.tight_layout()

# Salvando a figura do mapa de calor
figura_correlacao.savefig(
    "outputs/figures/matriz_correlacao.png",
    dpi=180,
    bbox_inches="tight",
)

# 7) Boxplot da taxa de juros por classificação

# Criando a área de desenho do boxplot
figura_boxplot, eixo = plt.subplots(figsize=(10, 6))

# Ordenando as classificações de empréstimo de A até G
ordem_classificacao = sorted(base["classificacao_emprestimo"].unique())

# Comparando a distribuição da taxa de juros entre as classificações
sns.boxplot(
    data=base,
    x="classificacao_emprestimo",
    y="taxa_juros_emprestimo",
    order=ordem_classificacao,
    color="steelblue",
    ax=eixo,
)

# Inserindo o título e os rótulos dos eixos
eixo.set_title(
    "Distribuição da taxa de juros por classificação de empréstimo",
    fontweight="bold",
)
eixo.set_xlabel("Classificação do empréstimo")
eixo.set_ylabel("Taxa de juros (%)")

# Ajustando a figura do boxplot
figura_boxplot.tight_layout()

# Salvando a figura do boxplot
figura_boxplot.savefig(
    "outputs/figures/taxa_juros_por_classificacao.png",
    dpi=180,
    bbox_inches="tight",
)
