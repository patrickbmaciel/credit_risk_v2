# Credit Risk v2

## 1. Introdução

Este projeto, desenvolvido em Python, analisa e classifica risco de crédito. O objetivo é identificar clientes com maior probabilidade de inadimplência e comparar modelos que apoiem decisões de concessão de crédito.

Antecipar a inadimplência é relevante porque uma aprovação indevida pode gerar perda financeira, enquanto uma recusa indevida pode reduzir receita e prejudicar a experiência do cliente. Por isso, a avaliação considera tanto o desempenho global quanto os tipos de erro, com atenção especial ao falso negativo, que ocorre quando uma pessoa inadimplente é classificada como não inadimplente.

## 2. Metodologia

### Base de dados

Foi utilizado o conjunto público [Credit Risk Dataset, do Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), com 32.581 registros e 12 variáveis sobre perfil do cliente, características do empréstimo e histórico de crédito.

O primeiro script baixa o arquivo bruto, renomeia e recodifica as variáveis, remove registros com valores ausentes e outliers. A base processada contém 28.629 registros: 22.427 não inadimplentes e 6.202 inadimplentes, o que evidencia o desbalanceamento da classe positiva.

A variável-alvo é `status_emprestimo`:

- `0`: não inadimplente;
- `1`: inadimplente e classe positiva das métricas.

### Algoritmos

| Algoritmo | Descrição | Estratégia de balanceamento |
|---|---|---|
| Regressão Logística | Estima a probabilidade de inadimplência por uma relação linear entre as variáveis e o logaritmo das chances. | Pesos de classe balanceados. |
| Árvore de Decisão | Cria regras sucessivas de divisão para separar inadimplentes e não inadimplentes. | Pesos de classe balanceados. |
| Gaussian Naive Bayes | Aplica o teorema de Bayes, assumindo independência condicional e distribuição normal das variáveis. | Pesos amostrais calculados no treino. |
| KNN (`k=5`) | Classifica cada caso pelo voto dos cinco vizinhos mais próximos. | Sobreamostragem aleatória apenas no treino. |
| Random Forest (100 árvores) | Combina várias árvores para reduzir a variância e produzir previsões mais robustas. | Pesos de classe balanceados. |

Cada algoritmo foi treinado em duas versões: original e balanceada. Os dados foram divididos uma única vez, de forma estratificada, em treino (70%) e teste (30%), com `random_state=123`. As dez versões foram avaliadas no mesmo teste desbalanceado, com 8.589 registros.

As variáveis categóricas foram transformadas por one-hot encoding, com a primeira categoria como referência. No KNN, as variáveis numéricas também foram normalizadas com Min-Max. Codificadores, escaladores, pesos e sobreamostragem foram ajustados ou aplicados somente no treino, evitando vazamento de dados. O limiar de classificação utilizado foi 0,5.

### Pipeline

O pipeline é composto por três etapas sequenciais:

| Ordem | Script | Responsabilidade | Principais artefatos |
|---:|---|---|---|
| 1 | `src/1_coleta_dados.py` | Baixar, validar, tratar e salvar os dados. | Base bruta, base processada, resumo do tratamento e estatísticas descritivas. |
| 2 | `src/2_analise_exploratoria.py` | Explorar exclusivamente a base processada. | Frequências, correlações e quatro figuras exploratórias. |
| 3 | `src/3_modelagens.py` | Preparar os dados, treinar dez versões e avaliá-las. | Métricas, coeficientes, importâncias, matrizes de confusão, curvas ROC e árvores. |

Para instalar as dependências e executar o projeto a partir da raiz:

```powershell
python -m pip install -r requirements.txt
python src/1_coleta_dados.py
python src/2_analise_exploratoria.py
python src/3_modelagens.py
```

## 3. Resultados

### 3.1 Análise exploratória dos dados

O tratamento removeu 3.943 registros com valores ausentes e outros nove pelos filtros de idade e tempo de emprego, totalizando 3.952 exclusões. As distribuições de valor do empréstimo, percentual da renda comprometida e duração do histórico de crédito apresentam assimetria à direita, com concentração nos valores mais baixos.

![Distribuições das variáveis relacionadas aos empréstimos](outputs/figures/distribuicoes_emprestimos.png)

Aluguel (50,81%) e hipoteca (41,21%) concentram 92,02% dos tipos de residência. As intenções de empréstimo são relativamente distribuídas, enquanto as classificações A e B representam 64,78% da base, enquanto as classes E, F e G são pouco frequentes.

![Distribuições das variáveis categóricas](outputs/figures/distribuicoes_categoricas.png)

A correlação mais forte ocorre entre idade e duração do histórico de crédito (0,88). Também se destacam a relação positiva entre valor do empréstimo e percentual da renda comprometida (0,58) e a relação negativa entre renda anual e percentual comprometido (-0,30).

![Matriz de correlação entre as variáveis](outputs/figures/matriz_correlacao.png)

As taxas de juros cobradas crescem de forma consistente das classificações A a G, indicando que classificações de maior risco estão associadas a juros mais elevados.

![Taxa de juros por classificação do empréstimo](outputs/figures/taxa_juros_por_classificacao.png)

### 3.2 Resultados dos modelos

- Regressão Logística: a versão original apresenta recall baixo (25,90%). O balanceamento eleva o recall para 77,38% e reduz os falsos negativos de 1.379 para 421, mas aumenta os falsos positivos de 124 para 1.961.
- Árvore de Decisão: as duas versões têm desempenho semelhante. O balanceamento melhora discretamente acurácia, precisão, F1-score e ROC AUC, mas aumenta os falsos negativos de 451 para 458.
- Gaussian Naive Bayes: o balanceamento eleva o recall de 27,14% para 74,48% e reduz os falsos negativos de 1.356 para 475. Em contrapartida, a precisão cai para 38,17% e os falsos positivos chegam a 2.245; a ROC AUC permanece em 78,44%.
- KNN: a versão original oferece maior precisão e F1-score. A versão balanceada reduz os falsos negativos de 717 para 459 e eleva o recall para 75,34%, com perda relevante de precisão, acurácia e ROC AUC.
- Random Forest: as duas versões apresentam o melhor desempenho global. A balanceada lidera acurácia, precisão e ROC AUC, mas a original tem maior recall, maior F1-score e menos falsos negativos.

Nas Árvores de Decisão e Random Forests, as variáveis mais importantes são, de forma consistente, `percentual_renda_emprestimo`, `taxa_juros_emprestimo` e `renda_anual`. As importâncias medem a redução de impureza usada nas divisões e não indicam direção do efeito nem causalidade.

Mais detalhes:
<details>
<summary>Regressão Logística: matriz de confusão e curva ROC</summary>

![Matrizes de confusão da Regressão Logística](outputs/figures/matriz_confusao_regressao_logistica.png)

![Curvas ROC da Regressão Logística](outputs/figures/curva_roc_regressao_logistica.png)

</details>

<details>
<summary>Árvore de Decisão: matriz de confusão e curva ROC</summary>

![Matrizes de confusão da Árvore de Decisão](outputs/figures/matriz_confusao_arvore_decisao.png)

![Curvas ROC da Árvore de Decisão](outputs/figures/curva_roc_arvore_decisao.png)

</details>

<details>
<summary>Gaussian Naive Bayes: matriz de confusão e curva ROC</summary>

![Matrizes de confusão do Gaussian Naive Bayes](outputs/figures/matriz_confusao_naive_bayes.png)

![Curvas ROC do Gaussian Naive Bayes](outputs/figures/curva_roc_naive_bayes.png)

</details>

<details>
<summary>KNN: matriz de confusão e curva ROC</summary>

![Matrizes de confusão do KNN](outputs/figures/matriz_confusao_knn.png)

![Curvas ROC do KNN](outputs/figures/curva_roc_knn.png)

</details>

<details>
<summary>Random Forest: matriz de confusão e curva ROC</summary>

![Matrizes de confusão da Random Forest](outputs/figures/matriz_confusao_random_forest.png)

![Curvas ROC da Random Forest](outputs/figures/curva_roc_random_forest.png)

</details>

<details>
<summary>Árvores de Decisão: três primeiros níveis</summary>

![Comparação das Árvores de Decisão](outputs/figures/arvores_decisao.png)

</details>

### 3.3 Avaliação dos modelos

As métricas abaixo foram calculadas no mesmo conjunto de teste.

| Modelo | Estratégia | FP | FN | Acurácia | Precisão | Recall | F1-score | ROC AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | Original | 124 | 1.379 | 82,50% | 79,54% | 25,90% | 39,08% | 79,21% |
| Regressão Logística | Balanceada | 1.961 | **421** | 72,27% | 42,34% | **77,38%** | 54,73% | 82,08% |
| Árvore de Decisão | Original | 480 | 451 | 89,16% | 74,60% | 75,77% | 75,18% | 84,32% |
| Árvore de Decisão | Balanceada | 447 | 458 | 89,46% | 75,84% | 75,39% | 75,61% | 84,37% |
| Gaussian Naive Bayes | Original | 209 | 1.356 | 81,78% | 70,73% | 27,14% | 39,22% | 78,44% |
| Gaussian Naive Bayes | Balanceada | 2.245 | 475 | 68,33% | 38,17% | 74,48% | 50,47% | 78,44% |
| KNN | Original | 186 | 717 | 89,49% | 86,02% | 61,47% | 71,70% | 85,94% |
| KNN | Balanceada | 1.203 | 459 | 80,65% | 53,82% | 75,34% | 62,79% | 84,42% |
| Random Forest | Original | 75 | 505 | 93,25% | 94,76% | 72,86% | **82,38%** | 93,20% |
| Random Forest | Balanceada | **56** | 520 | **93,29%** | **95,99%** | 72,06% | 82,32% | **93,48%** |

O Random Forest balanceado apresenta a maior acurácia, precisão e ROC AUC, porém aumenta os falsos negativos de 505 para 520 e reduz o recall de 72,86% para 72,06% em relação à versão original. Portanto, ela não é a melhor escolha quando o erro prioritário é deixar de identificar um inadimplente.

#### Estratégia champion–challenger

- Champion: o Random Forest original possui o maior F1-score (82,38%), ROC AUC de 93,20%, precisão de 94,76% e menos falsos negativos que a versão balanceada. É a melhor opção para desempenho global sem ampliar o erro mais custoso dentro da família de melhor desempenho.
- Challenger: a Regressão Logística balanceada tem o menor número de falsos negativos (421) e maior recall (77,38%). Deve ser testada como alternativa de maior sensibilidade, reconhecendo o custo de 1.961 falsos positivos e da precisão de 42,34%.

A comparação operacional entre champion e challenger deve usar uma matriz de custos. A adoção do challenger depende de o custo evitado pelos 84 falsos negativos a menos compensar os 1.886 falsos positivos adicionais em relação ao champion.

## 4. Considerações

Os resultados mostram que não existe um único modelo superior em todos os critérios. O Random Forest original oferece o melhor equilíbrio global, enquanto a Regressão Logística balanceada identifica uma proporção maior de inadimplentes. A decisão final deve refletir o custo financeiro de cada tipo de erro, e não apenas a acurácia.

Nesse cenário, recomenda-se acompanhar especialmente o percentual da renda comprometida, a taxa de juros e a renda anual, além de criar uma faixa de revisão manual para casos próximos ao limiar de decisão. Essas variáveis são úteis para priorização analítica, mas não devem ser interpretadas isoladamente nem como causas da inadimplência.

Entre os próximos passos mapeados, estão:

- definir uma matriz financeira de custos e otimizar o limiar de classificação;
- aplicar validação cruzada e validação temporal, além de ajustar hiperparâmetros;
- avaliar calibração das probabilidades, explicabilidade e desempenho por subgrupos;
- testar os modelos em dados externos e monitorar estabilidade e degradação ao longo do tempo.

As principais limitações são o uso de um dataset público sem validação externa, a exclusão completa dos registros com valores ausentes, a avaliação em uma única divisão treino-teste, o limiar fixo de 0,5 e a ausência de custos financeiros formalizados. Além disso, as importâncias por redução de impureza podem favorecer determinadas variáveis e não representam efeitos causais.
