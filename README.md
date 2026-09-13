# Credit Risk v2

## 1. Introdução

Este projeto, desenvolvido em Python, analisa e classifica risco de crédito. O objetivo é identificar clientes com maior probabilidade de inadimplência e comparar o desempenho e o resultado financeiro de modelos que apoiem decisões de concessão de crédito.

Antecipar a inadimplência é relevante porque uma aprovação indevida pode gerar perda financeira, enquanto uma recusa indevida pode reduzir receita e prejudicar a experiência do cliente. Por isso, a avaliação considera tanto o desempenho global quanto os tipos de erro, com atenção especial ao falso negativo, que ocorre quando uma pessoa inadimplente é classificada como não inadimplente.

## 2. Metodologia

### 2.1 Base de dados

Foi utilizado o conjunto público [Credit Risk Dataset, do Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), com 32.581 registros e 12 variáveis sobre perfil do cliente, características do empréstimo e histórico de crédito.

O primeiro script baixa o arquivo bruto, renomeia e recodifica as variáveis, remove registros com valores ausentes e outliers. A base processada contém 28.629 registros: 22.427 não inadimplentes e 6.202 inadimplentes, o que evidencia o desbalanceamento da classe positiva.

A variável-alvo é `status_emprestimo`:

- `0`: não inadimplente;
- `1`: inadimplente e classe positiva das métricas.

### 2.2 Algoritmos

| Algoritmo | Descrição | Estratégia de balanceamento |
|---|---|---|
| Regressão Logística | Estima a probabilidade de inadimplência por uma relação linear entre as variáveis e o logaritmo das chances. | Pesos de classe balanceados. |
| Árvore de Decisão | Cria regras sucessivas de divisão para separar inadimplentes e não inadimplentes. | Pesos de classe balanceados. |
| Gaussian Naive Bayes | Aplica o teorema de Bayes, assumindo independência condicional e distribuição normal das variáveis. | Pesos amostrais calculados no treino. |
| KNN (`k=5`) | Classifica cada caso pelo voto dos cinco vizinhos mais próximos. | Sobreamostragem aleatória apenas no treino. |
| Random Forest (100 árvores) | Combina várias árvores para reduzir a variância e produzir previsões mais robustas. | Pesos de classe balanceados. |
| XGBoost (100 árvores) | Constrói árvores sequencialmente, corrigindo os erros cometidos pelas anteriores. | Pesos amostrais calculados no treino. |
| LightGBM (100 árvores) | Utiliza gradient boosting com crescimento eficiente das árvores. | Pesos amostrais calculados no treino. |

Cada algoritmo foi treinado em duas versões: original e balanceada. Os dados foram divididos uma única vez, de forma estratificada, em treino (70%) e teste (30%), com `random_state=123`. As 14 versões foram avaliadas no mesmo teste desbalanceado, com 8.589 registros.

As variáveis categóricas foram transformadas por one-hot encoding, com a primeira categoria como referência. No KNN, as variáveis numéricas também foram normalizadas com Min-Max. Codificadores, escaladores, pesos e sobreamostragem foram ajustados ou aplicados somente no treino, evitando vazamento de dados. O limiar de classificação utilizado foi 0,5.

Além das métricas calculadas no limiar de 0,5 e da ROC AUC, a avaliação inclui a PR-AUC. Essa métrica resume a relação entre precisão e recall para diferentes limiares e é especialmente útil quando a classe positiva é menos frequente. Foi utilizada a Average Precision do scikit-learn, sem interpolação trapezoidal. Como referência, a proporção de inadimplentes no teste é de 21,67%.

### 2.3 Avaliação financeira

O lucro estimado considera somente os empréstimos previstos como não inadimplentes, que seriam aprovados pelo modelo. Para um cliente realmente adimplente, o resultado corresponde ao valor do empréstimo multiplicado pela taxa de juros. Para um cliente inadimplente aprovado, considera-se a perda integral do principal. Empréstimos previstos como inadimplentes são recusados e têm resultado financeiro igual a zero.

Essa é uma estimativa de um único período, em dólares. O cálculo não considera prazo, amortização, recuperação após inadimplência, custo de capital, despesas operacionais ou custo de oportunidade das recusas.

### 2.4 Pipeline

O pipeline é composto por três etapas sequenciais:

| Ordem | Script | Responsabilidade | Principais artefatos |
|---:|---|---|---|
| 1 | `src/1_coleta_dados.py` | Baixar, validar, tratar e salvar os dados. | Base bruta, base processada, resumo do tratamento e estatísticas descritivas. |
| 2 | `src/2_analise_exploratoria.py` | Explorar exclusivamente a base processada. | Frequências, correlações e quatro figuras exploratórias. |
| 3 | `src/3_modelagens.py` | Preparar os dados, treinar 14 versões e avaliá-las. | Métricas, lucro, seleção champion-challenger, coeficientes, importâncias, gráfico de importâncias, matrizes de confusão, curvas ROC, curvas precisão-recall, comparação gráfica do lucro e árvores. |

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
- Árvore de Decisão: as duas versões têm desempenho semelhante. O balanceamento melhora discretamente acurácia, precisão, F1-score, ROC AUC e lucro, mas aumenta os falsos negativos de 451 para 458.
- Gaussian Naive Bayes: o balanceamento eleva o recall de 27,14% para 74,48% e reduz os falsos negativos de 1.356 para 475. Em contrapartida, a precisão cai para 38,17% e os falsos positivos chegam a 2.245; a ROC AUC permanece em 78,44%.
- KNN: a versão original oferece maior precisão e F1-score. A versão balanceada reduz os falsos negativos de 717 para 459 e eleva o recall para 75,34%, com perda relevante de precisão, acurácia e ROC AUC.
- Random Forest: a versão original tem maior acurácia, precisão, F1-score e ROC AUC, enquanto a balanceada reduz os falsos negativos de 505 para 456 e aumenta o lucro de 2.209.972,85 para 2.540.361,69.
- XGBoost: a versão original alcança acurácia de 93,54% e ROC AUC de 94,55%. A balanceada reduz os falsos negativos de 493 para 377, obtém recall de 79,74% e apresenta o maior lucro estimado, de 3.031.737,10.
- LightGBM: a versão original registra a maior precisão (96,90%), a maior ROC AUC (94,85%) e a maior PR-AUC (90,25%) da comparação. A balanceada reduz os falsos negativos de 518 para 403 e alcança lucro de 2.768.269,27.

Nas Árvores de Decisão e Random Forests, destacam-se `percentual_renda_emprestimo`, `taxa_juros_emprestimo` e `renda_anual`. No XGBoost, também aparecem categorias de residência e classificação do empréstimo, enquanto no LightGBM se destacam renda, taxa de juros e valor do empréstimo. As importâncias são calculadas de forma e em escalas diferentes entre os algoritmos; devem ser comparadas dentro de cada modelo e não indicam direção do efeito nem causalidade.

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
<summary>XGBoost: matriz de confusão e curva ROC</summary>

![Matrizes de confusão do XGBoost](outputs/figures/matriz_confusao_xgboost.png)

![Curvas ROC do XGBoost](outputs/figures/curva_roc_xgboost.png)

</details>

<details>
<summary>LightGBM: matriz de confusão e curva ROC</summary>

![Matrizes de confusão do LightGBM](outputs/figures/matriz_confusao_lightgbm.png)

![Curvas ROC do LightGBM](outputs/figures/curva_roc_lightgbm.png)

</details>

<details>
<summary>Árvores de Decisão: três primeiros níveis</summary>

![Comparação das Árvores de Decisão](outputs/figures/arvores_decisao.png)

</details>

### 3.3 Curvas precisão-recall comparativas

A curva precisão-recall avalia o desempenho do modelo em diferentes limiares de classificação. Ela relaciona a proporção de inadimplentes corretamente identificados (recall) à proporção de clientes classificados como inadimplentes que realmente pertencem a essa classe (precisão). Essa análise é especialmente importante quando a classe de interesse é menos frequente, pois evidencia o equilíbrio entre identificar mais inadimplentes e evitar que clientes adimplentes sejam classificados incorretamente. Quanto mais a curva permanecer próxima da parte superior do gráfico, melhor é o desempenho do modelo.

Os resultados mostram que o efeito do balanceamento varia entre os algoritmos. A PR-AUC, que resume o desempenho do modelo ao longo da curva precisão–recall, aumenta nas Regressões Logísticas e nas Árvores de Decisão, apresenta uma melhora discreta no XGBoost, permanece inalterada no Naive Bayes e diminui no KNN, no Random Forest e no LightGBM. Apesar dessas diferenças, os modelos de boosting e o Random Forest continuam entre os melhores nas estratégias original e balanceada.

Como o balanceamento atribui maior importância à classe inadimplente durante o treinamento, o modelo tende a identificar mais inadimplentes no limiar de 0,5. Porém, isso não garante uma PR-AUC maior. Métricas como precisão, recall e falsos negativos são calculadas para um limiar específico, enquanto a PR-AUC avalia a capacidade do modelo de ordenar os clientes por risco considerando diferentes limiares. Assim, um modelo pode aumentar o recall em 0,5 e, ao mesmo tempo, apresentar uma PR-AUC menor caso essa mudança reduza sua precisão ou prejudique a ordenação geral das probabilidades.

![Curvas precisão-recall separadas por estratégia](outputs/figures/curvas_pr_por_estrategia.png)

### 3.4 Comparação do lucro estimado

O gráfico a seguir conecta os lucros das estratégias original e balanceada de cada algoritmo, permitindo observar tanto a direção quanto a magnitude da mudança. O balanceamento aumenta o lucro estimado em todos os modelos avaliados. O XGBoost balanceado apresenta o maior resultado, com 3,03 milhões de dólares, seguido pelo LightGBM balanceado, com 2,77 milhões, e pelo Random Forest balanceado, com 2,54 milhões.

A mudança mais expressiva ocorre na Regressão Logística, cujo lucro passa de aproximadamente -7,57 milhões para 0,34 milhão de dólares. O Gaussian Naive Bayes também reduz substancialmente a perda estimada, de -6,72 milhões para -0,39 milhão, mas permanece com resultado negativo. Essas diferenças mostram que o efeito financeiro do balanceamento depende do comportamento de cada algoritmo e da combinação entre empréstimos aprovados, juros recebidos e perdas com inadimplência.

![Efeito da estratégia no lucro estimado](outputs/figures/lucro_pontos_conectados.png)

### 3.5 Avaliação dos modelos

As métricas e os lucros abaixo foram calculados no mesmo conjunto de teste. FP representa um cliente adimplente recusado, enquanto FN representa um cliente inadimplente aprovado. A PR-AUC foi calculada a partir das probabilidades da classe inadimplente.

| Modelo | Estratégia | FP | FN | Lucro estimado ($) | Acurácia | Precisão | Recall | F1-score | ROC AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | Original | 124 | 1.379 | -7.570.532,98 | 82,50% | 79,54% | 25,90% | 39,08% | 79,21% | 58,34% |
| Regressão Logística | Balanceada | 1.961 | 421 | 342.134,76 | 72,27% | 42,34% | 77,38% | 54,73% | 82,08% | 59,79% |
| Árvore de Decisão | Original | 480 | 451 | 2.212.771,58 | 89,16% | 74,60% | 75,77% | 75,18% | 84,32% | 61,77% |
| Árvore de Decisão | Balanceada | 438 | 458 | 2.295.423,35 | 89,57% | 76,21% | 75,39% | 75,80% | 84,44% | 62,79% |
| Gaussian Naive Bayes | Original | 209 | 1.356 | -6.722.119,54 | 81,78% | 70,73% | 27,14% | 39,22% | 78,44% | 53,72% |
| Gaussian Naive Bayes | Balanceada | 2.245 | 475 | -386.512,82 | 68,33% | 38,17% | 74,48% | 50,47% | 78,44% | 53,72% |
| KNN | Original | 186 | 717 | 82.677,71 | 89,49% | 86,02% | 61,47% | 71,70% | 85,94% | 73,93% |
| KNN | Balanceada | 1.203 | 459 | 1.375.092,91 | 80,65% | 53,82% | 75,34% | 62,79% | 84,42% | 64,06% |
| Random Forest | Original | 75 | 505 | 2.209.972,85 | 93,25% | 94,76% | 72,86% | 82,38% | 93,20% | 88,03% |
| Random Forest | Balanceada | 178 | 456 | 2.540.361,69 | 92,62% | 88,76% | 75,50% | 81,59% | 93,33% | 87,89% |
| XGBoost | Original | 62 | 493 | 2.388.590,97 | **93,54%** | 95,66% | 73,51% | **83,14%** | 94,55% | 90,00% |
| XGBoost | Balanceada | 281 | **377** | **3.031.737,10** | 92,34% | 84,08% | **79,74%** | 81,85% | 94,47% | 90,06% |
| LightGBM | Original | **43** | 518 | 2.142.829,68 | 93,47% | **96,90%** | 72,17% | 82,72% | **94,85%** | **90,25%** |
| LightGBM | Balanceada | 326 | 403 | 2.768.269,27 | 91,51% | 81,73% | 78,34% | 80,00% | 94,50% | 89,70% |

O XGBoost balanceado apresenta simultaneamente o maior lucro, o menor número de falsos negativos e o maior recall. O LightGBM original lidera precisão, ROC AUC e PR-AUC, mas seu lucro é menor e ele deixa mais inadimplentes sem identificação. Sua PR-AUC de 90,25% é seguida de perto pelos 90,06% do XGBoost balanceado, ambos muito acima da prevalência de 21,67% da classe positiva.

### 3.6 Seleção dos modelos

Para organizar a seleção dos modelos, adota-se a estratégia champion-challenger (campeão-desafiante). Nessa abordagem, o modelo champion funciona como referência, enquanto o challenger é mantido como alternativa para comparação, monitoramento e eventual substituição.

Sendo assim:

- Champion: o XGBoost balanceado maximiza o lucro estimado, com 3,03 milhões de dólares. Também registra 377 falsos negativos, recall de 79,74%, ROC AUC de 94,47% e PR-AUC de 90,06%.
- Challenger: excluído o champion, o LightGBM balanceado apresenta o menor número de falsos negativos, com 403 casos. Seu lucro estimado é de 2,77 milhões de dólares, com recall de 78,34%, ROC AUC de 94,50% e PR-AUC de 89,70%.

O champion supera o challenger em 263.467,83 dólares e registra 26 falsos negativos a menos. O challenger permanece como alternativa distinta para comparação, contingência e monitoramento do desempenho fora da amostra.

### 3.7 Importâncias das variáveis

O gráfico abaixo apresenta as dez variáveis mais importantes do XGBoost balanceado, modelo selecionado como champion. As principais são o percentual da renda comprometida com o empréstimo, a posse de imóvel próprio e a classificação de crédito C. Elas podem refletir a capacidade de pagamento, a estabilidade patrimonial e o risco de crédito do cliente.

Também se destacam a finalidade do empréstimo, a existência de hipoteca e a taxa de juros. A feature importance mostra a contribuição das variáveis para o modelo, mas não indicam a direção do efeito nem comprovam causalidade.

![Importâncias das variáveis do XGBoost balanceado](outputs/figures/importancias_xgboost_balanceado_top10.png)

### 3.8 Interpretabilidade do modelo

O SHAP aprofunda a feature importance ao apresentar não apenas quais variáveis mais influenciam o XGBoost balanceado, mas também a direção e a intensidade de seus efeitos. Cada ponto representa um empréstimo do conjunto de teste. Valores SHAP positivos aumentam o risco de inadimplência estimado pelo modelo, enquanto valores negativos o reduzem.

As cores representam o valor observado da variável em cada cliente. Rosa indica valores mais altosa, enquanto azul indica valores mais baixos. Em variáveis binárias, como possuir imóvel próprio, rosa representa a presença da característica e azul sua ausência. A posição horizontal mostra o efeito sobre a previsão, à direita aumenta o risco estimado e à esquerda o reduz.

A taxa de juros, a renda anual e o percentual da renda comprometida com o empréstimo apresentam os maiores impactos médios. Taxas de juros e percentuais de comprometimento mais altos tendem a elevar o risco previsto, enquanto rendas mais altas tendem a reduzi-lo. Também se observa que a classificação de crédito D aumenta o risco estimado. Imóvel próprio, hipoteca e empréstimos destinados a empreendimento ou educação aparecem associados à redução do risco.

A ordem difere da feature importance porque as duas medidas respondem a perguntas distintas. A importância resume o uso das variáveis nas árvores, enquanto o SHAP mede o impacto de cada variável sobre as previsões individuais. Ambas não representam causalidade.

![SHAP do XGBoost balanceado](outputs/figures/shap_xgboost_balanceado.png)

## 4. Considerações

Os resultados mostram que não existe um único modelo superior em todos os critérios. O XGBoost balanceado foi escolhido como champion porque maximiza o lucro estimado e, nesta amostra, também identifica a maior proporção de inadimplentes. O LightGBM original lidera precisão, ROC AUC e PR-AUC. A decisão final deve refletir o resultado financeiro e os tipos de erro, e não apenas a acurácia.

Nesse cenário, recomenda-se acompanhar especialmente o percentual da renda comprometida, a taxa de juros e a renda anual, além de criar uma faixa de revisão manual para casos próximos ao limiar de decisão. Essas variáveis são úteis para priorização analítica, mas não devem ser interpretadas isoladamente nem como causas da inadimplência.

Entre os próximos passos mapeados, estão:

- incorporar prazo, recuperação, custo de capital e despesas à avaliação financeira;
- otimizar o limiar de classificação com base no lucro e nos falsos negativos;
- aplicar validação cruzada e validação temporal, além de ajustar hiperparâmetros;
- avaliar calibração das probabilidades, explicabilidade e desempenho por subgrupos;
- testar os modelos em dados externos e monitorar estabilidade e degradação ao longo do tempo.

As principais limitações são o uso de um dataset público sem validação externa, a exclusão completa dos registros com valores ausentes, a avaliação e a seleção em uma única divisão treino-teste e o limiar fixo de 0,5. O lucro é uma aproximação de um período com perda integral em caso de inadimplência e sem outros custos ou recuperações. Além disso, as importâncias nativas podem favorecer determinadas variáveis, não são diretamente comparáveis entre algoritmos e não representam efeitos causais.
