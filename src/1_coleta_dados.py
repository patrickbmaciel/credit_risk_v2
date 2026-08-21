# 0) Configurações iniciais

# Importando pacotes
from pathlib import Path
import kagglehub
import pandas as pd

# Definindo os caminhos relativos usados no script
diretorio_raw = "data/raw"
diretorio_processed = "data/processed"
diretorio_tabelas = "outputs/tables"
arquivo_bruto = "data/raw/credit_risk_dataset.csv"
arquivo_processado = "data/processed/credit_risk_processed.csv"

# Definindo o conjunto de dados público que será baixado do Kaggle
identificador_kaggle = "laotse/credit-risk-dataset"

# Listando as 12 colunas esperadas no arquivo original
colunas_brutas = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

# Relacionando os nomes originais aos nomes em português
renomear_colunas = {
    "person_age": "idade",
    "person_income": "renda_anual",
    "person_home_ownership": "residencia",
    "person_emp_length": "tempo_emprego",
    "loan_intent": "intencao_emprestimo",
    "loan_grade": "classificacao_emprestimo",
    "loan_amnt": "valor_emprestimo",
    "loan_int_rate": "taxa_juros_emprestimo",
    "loan_status": "status_emprestimo",
    "loan_percent_income": "percentual_renda_emprestimo",
    "cb_person_default_on_file": "historico_inadimplencia",
    "cb_person_cred_hist_length": "duracao_historico_credito",
}

# Relacionando as categorias de residência aos nomes em português
mapa_residencia = {
    "RENT": "aluguel",
    "OWN": "propria",
    "MORTGAGE": "hipoteca",
    "OTHER": "outro",
}

# Relacionando as intenções de empréstimo aos nomes em português
mapa_intencao = {
    "PERSONAL": "pessoal",
    "EDUCATION": "educacao",
    "MEDICAL": "medico",
    "VENTURE": "empreendimento",
    "HOMEIMPROVEMENT": "melhoria_residencial",
    "DEBTCONSOLIDATION": "consolidacao_dividas",
}

# Convertendo o histórico de inadimplência de texto para número
mapa_inadimplencia = {"Y": 1, "N": 0}

# Definindo as classificações de empréstimo aceitas
classificacoes_esperadas = set("ABCDEFG")

# Listando as variáveis usadas nas estatísticas descritivas
variaveis_numericas = [
    "idade",
    "renda_anual",
    "tempo_emprego",
    "valor_emprestimo",
    "taxa_juros_emprestimo",
    "percentual_renda_emprestimo",
    "duracao_historico_credito",
]

# 1) Coleta da base de dados

# Baixando novamente o CSV do Kaggle em toda execução do script
caminho_download = kagglehub.dataset_download(
    identificador_kaggle,
    output_dir=diretorio_raw,
    force_download=True,
)

# Lendo diretamente o arquivo bruto salvo em data/raw
base_bruta = pd.read_csv(arquivo_bruto)

# 2) Validação da base bruta

# Identificando colunas esperadas que não aparecem no arquivo
colunas_ausentes = sorted(set(colunas_brutas) - set(base_bruta.columns))

# Identificando colunas inesperadas que aparecem no arquivo
colunas_extras = sorted(set(base_bruta.columns) - set(colunas_brutas))

# Mantendo as colunas na mesma ordem definida para o projeto
base_bruta = base_bruta[colunas_brutas].copy()

# Identificando categorias desconhecidas na variável de residência
residencias_observadas = set(
    base_bruta["person_home_ownership"].dropna().astype(str).unique()
)
residencias_desconhecidas = residencias_observadas - set(mapa_residencia)

# Identificando categorias desconhecidas na intenção do empréstimo
intencoes_observadas = set(
    base_bruta["loan_intent"].dropna().astype(str).unique()
)
intencoes_desconhecidas = intencoes_observadas - set(mapa_intencao)

# Identificando categorias desconhecidas no histórico de inadimplência
historicos_observados = set(
    base_bruta["cb_person_default_on_file"].dropna().astype(str).unique()
)
historicos_desconhecidos = historicos_observados - set(mapa_inadimplencia)

# Identificando classificações de empréstimo desconhecidas
classificacoes_observadas = set(
    base_bruta["loan_grade"].dropna().astype(str).unique()
)
classificacoes_desconhecidas = (
    classificacoes_observadas - classificacoes_esperadas
)

# 3) Renomeação e recodificação das variáveis

# Renomeando as 12 colunas para português
base_ajustada = base_bruta.rename(columns=renomear_colunas).copy()

# Recodificando as categorias de residência
base_ajustada["residencia"] = base_ajustada["residencia"].map(
    mapa_residencia
)

# Recodificando as categorias de intenção do empréstimo
base_ajustada["intencao_emprestimo"] = base_ajustada[
    "intencao_emprestimo"
].map(mapa_intencao)

# Recodificando o histórico de inadimplência como zero ou um
base_ajustada["historico_inadimplencia"] = base_ajustada[
    "historico_inadimplencia"
].map(mapa_inadimplencia)

# Colocando a variável alvo na primeira coluna da base
ordem_colunas = ["status_emprestimo"] + [
    coluna
    for coluna in base_ajustada.columns
    if coluna != "status_emprestimo"
]
base_ajustada = base_ajustada[ordem_colunas]

# 4) Estatísticas anteriores à limpeza

# O objetivo é registrar uma fotografia numérica da base antes da limpeza. 
# Sem essa tabela, ainda seria possível limpar os dados, mas ficaria mais
# difícil demonstrar quais problemas existiam e qual foi o efeito do tratamento.

# Calculando as estatísticas das variáveis numéricas antes da limpeza
estatisticas_antes = base_ajustada[variaveis_numericas].describe().T

# Transformando os nomes das variáveis em uma coluna da tabela
estatisticas_antes = estatisticas_antes.reset_index(names="variavel")

# Mantendo somente as estatísticas utilizadas no projeto
colunas_estatisticas = [
    "variavel",
    "count",
    "mean",
    "std",
    "min",
    "25%",
    "50%",
    "75%",
    "max",
]
estatisticas_antes = estatisticas_antes[colunas_estatisticas]

# 5) Limpeza da base

# Removendo todas as observações que possuem valores ausentes
base_sem_ausentes = base_ajustada.dropna().copy()

# Removendo idades e tempos de emprego acima dos limites, ou seja, apenas 
# idades superiores a 77 ou tempos de emprego superiores a 65 são removidos:
# 1. A idade máxima registrada é de 144 anos, indicando provável outlier, 
# pois a expectativa de vida média é 77 anos, segundo o Centro de Controle
# de Doenças dos EUA;
# 2. O tempo de emprego de 123 anos indica erro, pois está associado a jovens
# na base de dados. Considerando a expectativa de vida e 12 anos de escolaridade, 
# o tempo máximo de trabalho poderia ser de 65 anos. Por outro lado, um tempo
# de emprego de 0 anos pode ser aceitável em alguns contextos, como para recém-formados.
base_tratada = base_sem_ausentes.loc[
    (base_sem_ausentes["idade"] <= 77)
    & (base_sem_ausentes["tempo_emprego"] <= 65)
].copy()

# Garantindo que a variável alvo seja representada por números inteiros
base_tratada["status_emprestimo"] = base_tratada[
    "status_emprestimo"
].astype(int)

# Garantindo que o histórico de inadimplência seja representado por inteiros
base_tratada["historico_inadimplencia"] = base_tratada[
    "historico_inadimplencia"
].astype(int)

# Recriando o índice depois da remoção de observações
base_tratada = base_tratada.reset_index(drop=True)

# 6) Estatísticas posteriores à limpeza

# Calculando as estatísticas das variáveis numéricas depois da limpeza
estatisticas_depois = base_tratada[variaveis_numericas].describe().T

# Transformando os nomes das variáveis em uma coluna da tabela
estatisticas_depois = estatisticas_depois.reset_index(names="variavel")

# Mantendo as mesmas estatísticas da tabela anterior
estatisticas_depois = estatisticas_depois[colunas_estatisticas]

# Registrando a quantidade de observações em cada etapa do tratamento
resumo_tratamento = pd.DataFrame(
    {
        "etapa": [
            "base_bruta",
            "apos_remocao_ausentes",
            "apos_filtros_outliers",
        ],
        "quantidade_registros": [
            len(base_bruta),
            len(base_sem_ausentes),
            len(base_tratada),
        ],
    }
)

# 7) Salvamento da base tratada e das tabelas auxiliares

# Salvando a base tratada que será usada pelos scripts subsequentes
base_tratada.to_csv(
    arquivo_processado,
    index=False,
    encoding="utf-8-sig",
)

# Salvando a quantidade de registros em cada etapa do tratamento
resumo_tratamento.to_csv(
    "outputs/tables/resumo_tratamento.csv",
    index=False,
    encoding="utf-8-sig",
)

# Salvando as estatísticas anteriores à limpeza
estatisticas_antes.to_csv(
    "outputs/tables/estatisticas_numericas_antes.csv",
    index=False,
    encoding="utf-8-sig",
)

# Salvando as estatísticas posteriores à limpeza
estatisticas_depois.to_csv(
    "outputs/tables/estatisticas_numericas_depois.csv",
    index=False,
    encoding="utf-8-sig",
)
