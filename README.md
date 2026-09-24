# Credit Risk v2

## 1. Introdução

Este projeto, desenvolvido em Python, analisa e classifica risco de crédito. O objetivo é identificar clientes com maior probabilidade de inadimplência e comparar o desempenho e o resultado financeiro de modelos que apoiem decisões de concessão de crédito.

Antecipar a inadimplência é relevante porque uma aprovação indevida pode gerar perda financeira, enquanto uma recusa indevida pode reduzir receita e prejudicar a experiência do cliente. Por isso, a avaliação considera tanto o desempenho global quanto os tipos de erro, com atenção especial ao falso negativo, que ocorre quando uma pessoa inadimplente é classificada como não inadimplente.

## 2. Metodologia

### 2.1 Base de dados

Foi utilizado o conjunto público [Credit Risk Dataset, do Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), com 32.581 registros e 12 variáveis sobre perfil do cliente, características do empréstimo e histórico de crédito.

O primeiro script baixa o arquivo bruto via API, renomeia e recodifica as variáveis, remove registros com valores ausentes e outliers. A base processada contém 28.629 registros: 22.427 não inadimplentes e 6.202 inadimplentes. A classe positiva representa 21,66% da amostra.

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

Cada algoritmo foi treinado em duas versões, original e balanceada, totalizando 14 candidatos. Os hiperparâmetros foram mantidos fixos e o limiar de classificação utilizado foi 0,5.

As variáveis categóricas foram transformadas por one-hot encoding, com a primeira categoria como referência. No KNN, as variáveis numéricas também foram normalizadas com Min-Max.

### 2.3 Nested cross-validation estratificado 10x5

A seleção e a avaliação foram realizadas por nested cross-validation (nested CV) estratificado 10x5, sem holdout separado. O número 10 representa os folds do ciclo externo, usados para avaliar o desempenho fora da amostra, enquanto o número 5 representa os folds do ciclo interno, usados para comparar os 14 candidatos e selecionar champion e challenger.

Em cada uma das dez rodadas externas, 90% dos registros formam o treino externo e os 10% restantes formam o teste externo. Dentro desses 90%, a validação interna de cinco folds realiza a seleção dos modelos sem consultar o teste externo. Depois da seleção interna, os candidatos são ajustados no treino externo e avaliados nos 10% reservados. A estratificação preserva aproximadamente a proporção de inadimplentes nos folds internos e externos. Codificadores, normalizadores, pesos e sobreamostragem são ajustados novamente e exclusivamente no treino de cada fold, evitando vazamento de dados.

Essa abordagem foi escolhida porque separa a seleção da avaliação e reduz a dependência de uma única divisão aleatória. Diferentemente do holdout 70/30, todos os registros participam da avaliação externa, cada um exatamente uma vez, e os resultados dos folds permitem examinar a estabilidade dos modelos. Em contrapartida, o nested CV exige mais ajustes e maior tempo de processamento.

O erro-padrão resume a incerteza da média obtida nos cinco folds internos e é calculado dividindo o desvio-padrão entre os folds pela raiz quadrada de cinco. O desvio-padrão mede quanto o desempenho varia entre os folds, enquanto o erro-padrão mede a precisão com que a média foi estimada.

O champion é definido pela regra de um erro-padrão aplicada ao lucro por registro. Primeiro, identifica-se o maior lucro médio e usa-se o erro-padrão desse candidato para estabelecer o limite de elegibilidade. Permanecem na comparação os modelos cuja média esteja até um erro-padrão abaixo da melhor média. Entre eles, é escolhido o candidato com menor desvio-padrão do lucro. Assim, uma pequena vantagem média, potencialmente decorrente da divisão dos dados, não prevalece automaticamente sobre um modelo mais estável. Os desempates consideram a menor taxa de falsos negativos e o maior lucro médio.

Depois da exclusão do champion, o challenger é definido de forma semelhante, mas com prioridade para a taxa de falsos negativos. São elegíveis os candidatos até um erro-padrão acima da menor taxa média de falsos negativos, e vence o candidato com menor dispersão dessa taxa. O lucro médio é usado como desempate.

Uma previsão out-of-fold (OOF) é produzida para um registro por um modelo que não utilizou esse registro no treinamento. Ao reunir os dez testes externos, obtém-se uma previsão OOF para cada um dos 28.629 registros e para cada candidato. Essas previsões são usadas nas métricas, matrizes de confusão e curvas finais. Após a avaliação externa, uma nova validação interna de cinco folds sobre toda a base escolhe os modelos finais, que são reajustados em todos os registros somente para gerar coeficientes, importâncias, SHAP e visualizações das árvores.

### 2.4 Métricas de desempenho

A inadimplência (`1`) é a classe positiva. Com o limiar de classificação de 0,5, probabilidades iguais ou superiores a esse valor são classificadas como inadimplência. A matriz de confusão organiza os quatro resultados possíveis:

| Resultado | Significado no projeto |
|---|---|
| Verdadeiro positivo (`TP`) | Cliente inadimplente corretamente identificado e empréstimo recusado. |
| Verdadeiro negativo (`TN`) | Cliente adimplente corretamente identificado e empréstimo aprovado. |
| Falso positivo (`FP`) | Cliente adimplente classificado como inadimplente e recusado incorretamente. |
| Falso negativo (`FN`) | Cliente inadimplente classificado como adimplente e aprovado incorretamente. |

As métricas calculadas a partir desses resultados e das probabilidades previstas são:

| Métrica | Fórmula ou cálculo | Interpretação | Direção desejável |
|---|---|---|---|
| Acurácia | `(TP + TN) / total` | Proporção de classificações corretas entre todos os registros. | Maior |
| Precisão | `TP / (TP + FP)` | Entre os clientes classificados como inadimplentes, proporção que realmente inadimpliu. | Maior |
| Recall | `TP / (TP + FN)` | Entre os clientes realmente inadimplentes, proporção identificada pelo modelo. | Maior |
| Taxa de falsos negativos | `FN / (FN + TP) = 1 − recall` | Proporção de inadimplentes não identificados e aprovados pelo modelo. | Menor |
| F1-score | `2 x (precisão x recall) / (precisão + recall)` | Média harmônica que resume o equilíbrio entre precisão e recall. | Maior |
| ROC AUC | Área sob a curva ROC | Capacidade de ordenar inadimplentes acima de adimplentes ao longo de diferentes limiares. | Maior |
| PR-AUC | Average Precision ao longo da curva precisão-recall | Resume o equilíbrio entre identificar inadimplentes e evitar classificações positivas incorretas em diferentes limiares. | Maior |

Acurácia, precisão, recall, taxa de falsos negativos e F1-score utilizam as classes definidas pelo limiar de 0,5. ROC AUC e PR-AUC utilizam diretamente as probabilidades e avaliam a ordenação dos clientes em diferentes limiares. No projeto, a PR-AUC corresponde à Average Precision calculada pelo `average_precision_score`.

A acurácia isolada pode ocultar desempenho ruim na classe minoritária. Em risco de crédito, recall elevado e taxa de falsos negativos reduzida ajudam a evitar a aprovação de inadimplentes, enquanto a precisão ajuda a controlar recusas indevidas de clientes adimplentes. Uma ordenação aleatória produz ROC AUC próxima de 0,5. Na PR-AUC, a referência aproxima-se da prevalência da classe positiva, que é de 21,66% nesta base. Por isso, as métricas devem ser interpretadas em conjunto e complementadas pela avaliação financeira.

### 2.5 Avaliação financeira

O lucro estimado considera somente os empréstimos previstos como não inadimplentes, que seriam aprovados pelo modelo. Para um cliente realmente adimplente, o resultado corresponde ao valor do empréstimo multiplicado pela taxa de juros. Para um cliente inadimplente aprovado, considera-se a perda integral do principal. Empréstimos previstos como inadimplentes são recusados e têm resultado financeiro igual a zero.

Essa é uma estimativa de um único período, em dólares. O cálculo não considera prazo, amortização, recuperação após inadimplência, custo de capital, despesas operacionais ou custo de oportunidade das recusas. Na seleção interna, utiliza-se o lucro por registro para tornar comparáveis folds de tamanhos ligeiramente diferentes. Nos resultados finais, o lucro total OOF cobre toda a base processada.

### 2.6 Pipeline

| Ordem | Script | Responsabilidade | Principais artefatos |
|---:|---|---|---|
| 1 | `src/1_coleta_dados.py` | Baixar, validar, tratar e salvar os dados. | Base bruta, base processada, resumo do tratamento e estatísticas descritivas. |
| 2 | `src/2_analise_exploratoria.py` | Explorar exclusivamente a base processada. | Frequências, correlações e quatro figuras exploratórias. |
| 3 | `src/3_modelagens_ncv.py` | Executar o nested CV 10x5, comparar 14 candidatos e reajustar os modelos finais. | Métricas OOF, resultados por fold, seleção champion-challenger, coeficientes, importâncias, SHAP e figuras de avaliação. |

Para instalar as dependências e executar o projeto a partir da raiz:

```powershell
python -m pip install -r requirements.txt
python src/1_coleta_dados.py
python src/2_analise_exploratoria.py
python src/3_modelagens_ncv.py
```

## 3. Resultados

### 3.1 Análise exploratória dos dados

O tratamento removeu 3.943 registros com valores ausentes e outros nove pelos filtros de idade e tempo de emprego, totalizando 3.952 exclusões. As distribuições de valor do empréstimo, percentual da renda comprometida e duração do histórico de crédito apresentam assimetria à direita, com concentração nos valores mais baixos.

![Distribuições das variáveis relacionadas aos empréstimos](outputs/figures/distribuicoes_emprestimos.png)

Aluguel (50,81%) e hipoteca (41,21%) concentram 92,02% dos tipos de residência. As intenções de empréstimo são relativamente distribuídas, enquanto as classificações A e B representam 64,78% da base e as classes E, F e G são pouco frequentes.

![Distribuições das variáveis categóricas](outputs/figures/distribuicoes_categoricas.png)

A correlação mais forte ocorre entre idade e duração do histórico de crédito (0,88). Também se destacam a relação positiva entre valor do empréstimo e percentual da renda comprometida (0,58) e a relação negativa entre renda anual e percentual comprometido (-0,30).

![Matriz de correlação entre as variáveis](outputs/figures/matriz_correlacao.png)

As taxas de juros cobradas crescem de forma consistente das classificações A a G, indicando que classificações de maior risco estão associadas a juros mais elevados.

![Taxa de juros por classificação do empréstimo](outputs/figures/taxa_juros_por_classificacao.png)

### 3.2 Resultados out-of-fold dos modelos

- Regressão Logística: o balanceamento eleva o recall de 22,80% para 77,14% e reduz os falsos negativos de 4.788 para 1.418. Em contrapartida, a precisão cai para 42,59% e os falsos positivos aumentam para 6.449.
- Árvore de Decisão: as duas versões apresentam resultados próximos. A original obtém maior recall, PR-AUC e lucro, enquanto a balanceada registra acurácia e precisão discretamente maiores.
- Gaussian Naive Bayes: o balanceamento aumenta o recall de 27,56% para 75,57% e reduz os falsos negativos, mas as duas versões apresentam lucro negativo. ROC AUC e PR-AUC permanecem praticamente inalteradas.
- KNN: a versão original tem maior acurácia, precisão, F1-score, ROC AUC e PR-AUC. A versão balanceada reduz os falsos negativos de 2.381 para 1.474 e transforma o lucro total de negativo para positivo.
- Random Forest: a versão original lidera acurácia, precisão, F1-score e PR-AUC dentro do algoritmo. A balanceada aumenta recall e lucro, alcançando 7,83 milhões de dólares.
- XGBoost: a versão original apresenta o maior F1-score, ROC AUC e PR-AUC da comparação. A balanceada reduz os falsos negativos para 1.241, atinge recall de 79,99% e produz o maior lucro OOF, de 9,62 milhões de dólares.
- LightGBM: a versão original alcança a maior acurácia e precisão. A balanceada reduz os falsos negativos para 1.264, obtém recall de 79,62% e lucro de 9,41 milhões de dólares.

As matrizes e curvas abaixo utilizam exclusivamente previsões externas OOF.

<details>
<summary>Regressão Logística: matriz de confusão e curva ROC</summary>

![Matrizes de confusão OOF da Regressão Logística](outputs/figures/matriz_confusao_regressao_logistica_ncv.png)

![Curvas ROC OOF da Regressão Logística](outputs/figures/curva_roc_regressao_logistica_ncv.png)

</details>

<details>
<summary>Árvore de Decisão: matriz de confusão e curva ROC</summary>

![Matrizes de confusão OOF da Árvore de Decisão](outputs/figures/matriz_confusao_arvore_decisao_ncv.png)

![Curvas ROC OOF da Árvore de Decisão](outputs/figures/curva_roc_arvore_decisao_ncv.png)

</details>

<details>
<summary>Gaussian Naive Bayes: matriz de confusão e curva ROC</summary>

![Matrizes de confusão OOF do Gaussian Naive Bayes](outputs/figures/matriz_confusao_naive_bayes_ncv.png)

![Curvas ROC OOF do Gaussian Naive Bayes](outputs/figures/curva_roc_naive_bayes_ncv.png)

</details>

<details>
<summary>KNN: matriz de confusão e curva ROC</summary>

![Matrizes de confusão OOF do KNN](outputs/figures/matriz_confusao_knn_ncv.png)

![Curvas ROC OOF do KNN](outputs/figures/curva_roc_knn_ncv.png)

</details>

<details>
<summary>Random Forest: matriz de confusão e curva ROC</summary>

![Matrizes de confusão OOF da Random Forest](outputs/figures/matriz_confusao_random_forest_ncv.png)

![Curvas ROC OOF da Random Forest](outputs/figures/curva_roc_random_forest_ncv.png)

</details>

<details>
<summary>XGBoost: matriz de confusão e curva ROC</summary>

![Matrizes de confusão OOF do XGBoost](outputs/figures/matriz_confusao_xgboost_ncv.png)

![Curvas ROC OOF do XGBoost](outputs/figures/curva_roc_xgboost_ncv.png)

</details>

<details>
<summary>LightGBM: matriz de confusão e curva ROC</summary>

![Matrizes de confusão OOF do LightGBM](outputs/figures/matriz_confusao_lightgbm_ncv.png)

![Curvas ROC OOF do LightGBM](outputs/figures/curva_roc_lightgbm_ncv.png)

</details>

<details>
<summary>Árvores de Decisão: três primeiros níveis do ajuste final</summary>

![Comparação das Árvores de Decisão reajustadas](outputs/figures/arvores_decisao_ncv.png)

</details>

### 3.3 Curvas precisão-recall comparativas

A curva precisão-recall relaciona a proporção de inadimplentes corretamente identificados à proporção de clientes classificados como inadimplentes que realmente pertencem a essa classe. Ela é especialmente informativa diante do desbalanceamento da variável-alvo.

Os modelos de boosting e a Random Forest apresentam as maiores PR-AUCs nas duas estratégias. O balanceamento aumenta o recall no limiar de 0,5, mas não necessariamente melhora a PR-AUC, pois essa métrica avalia a ordenação dos clientes ao longo de diferentes limiares.

![Curvas precisão-recall OOF por estratégia](outputs/figures/curvas_pr_por_estrategia_ncv.png)

![Curvas precisão-recall OOF dos modelos finais](outputs/figures/curvas_pr_modelos_destacados_ncv.png)

### 3.4 Comparação do lucro estimado

O balanceamento melhora substancialmente o resultado financeiro da Regressão Logística, do Gaussian Naive Bayes, do KNN, da Random Forest e dos modelos de boosting. Nas Árvores de Decisão, a versão original apresenta lucro superior.

O XGBoost balanceado registra o maior lucro OOF, com 9,62 milhões de dólares, equivalente a 335,90 dólares por registro. O LightGBM balanceado vem em seguida, com 9,41 milhões e 328,81 dólares por registro. Como os valores agregam previsões OOF para toda a base, não devem ser comparados diretamente aos lucros calculados anteriormente sobre um teste de 30%.

![Efeito da estratégia no lucro OOF](outputs/figures/lucro_pontos_conectados_ncv.png)

### 3.5 Avaliação dos modelos

FP representa um cliente adimplente recusado e FN representa um cliente inadimplente aprovado. Para cada candidato, as previsões dos dez testes externos são reunidas em um único conjunto com 28.629 previsões OOF. Cada registro aparece uma vez, sempre previsto por um modelo que não o utilizou no treinamento. As métricas da tabela são então calculadas uma única vez sobre esse conjunto completo. Por isso, as células de cada matriz de confusão somam 28.629 observações.

Essas métricas OOF agregadas representam o desempenho global por registro e não são médias simples das métricas dos dez folds. As médias e os desvios-padrão por fold têm outra função: mostram o desempenho típico e sua estabilidade entre diferentes amostras de teste. As duas leituras são complementares. A coluna de lucro por registro permite comparar a escala financeira independentemente do tamanho da amostra, enquanto o lucro OOF total, por abranger toda a base, não deve ser comparado diretamente ao lucro do antigo teste de 30%.

| Modelo | Estratégia | FP | FN | Lucro OOF ($) | Lucro/registro ($) | Acurácia | Precisão | Recall | F1-score | ROC AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | Original | 414 | 4.788 | -26.398.602,14 | -922,09 | 81,83% | 77,35% | 22,80% | 35,22% | 79,07% | 57,58% |
| Regressão Logística | Balanceada | 6.449 | 1.418 | 964.233,68 | 33,68 | 72,52% | 42,59% | 77,14% | 54,88% | 81,95% | 59,93% |
| Árvore de Decisão | Original | 1.714 | 1.446 | 7.023.847,86 | 245,34 | 88,96% | 73,51% | 76,68% | 75,06% | 84,52% | 61,42% |
| Árvore de Decisão | Balanceada | 1.555 | 1.519 | 6.414.297,71 | 224,05 | 89,26% | 75,07% | 75,51% | 75,29% | 84,29% | 61,99% |
| Gaussian Naive Bayes | Original | 746 | 4.493 | -23.259.707,55 | -812,45 | 81,70% | 69,61% | 27,56% | 39,48% | 77,98% | 53,28% |
| Gaussian Naive Bayes | Balanceada | 7.993 | 1.515 | -1.681.351,89 | -58,73 | 66,79% | 36,96% | 75,57% | 49,65% | 77,98% | 53,28% |
| KNN | Original | 645 | 2.381 | -90.147,45 | -3,15 | 89,43% | 85,56% | 61,61% | 71,63% | 86,51% | 74,67% |
| KNN | Balanceado | 3.894 | 1.474 | 4.960.090,81 | 173,25 | 81,25% | 54,84% | 76,23% | 63,79% | 85,16% | 65,50% |
| Random Forest | Original | 235 | 1.695 | 6.472.265,56 | 226,07 | 93,26% | 95,04% | 72,67% | 82,36% | 93,15% | 88,14% |
| Random Forest | Balanceada | 653 | 1.505 | 7.830.974,97 | 273,53 | 92,46% | 87,79% | 75,73% | 81,32% | 93,29% | 88,09% |
| XGBoost | Original | 234 | 1.631 | 7.354.210,02 | 256,88 | 93,49% | 95,13% | 73,70% | **83,06%** | **94,73%** | **90,28%** |
| XGBoost | Balanceado | 1.145 | **1.241** | **9.616.407,55** | **335,90** | 91,67% | 81,25% | **79,99%** | 80,61% | 94,71% | 90,17% |
| LightGBM | Original | **145** | 1.718 | 6.556.814,78 | 229,03 | **93,49%** | **96,87%** | 72,30% | 82,80% | 94,70% | 90,09% |
| LightGBM | Balanceado | 1.180 | 1.264 | 9.413.362,18 | 328,81 | 91,46% | 80,71% | 79,62% | 80,16% | 94,70% | 89,97% |

### 3.6 Seleção dos modelos

Nos dez ciclos externos, somente XGBoost balanceado e LightGBM balanceado foram selecionados. O LightGBM ocupou o papel de champion em sete folds e de challenger em três. O XGBoost foi champion em três e challenger em sete. Essa alternância mostra que os dois modelos apresentam desempenho próximo e que a regra de um erro-padrão evita definir o vencedor apenas pela maior média observada em uma partição.

Na validação interna final sobre toda a base:

- Champion: XGBoost balanceado. Nas previsões OOF, apresenta o maior lucro, de 9,62 milhões de dólares, o menor número de falsos negativos, com 1.241 casos, e o maior recall, de 79,99%. O lucro por registro entre os dez folds externos tem média de 335,90 dólares e desvio-padrão de 39,24.
- Challenger: LightGBM balanceado. Apresenta lucro OOF de 9,41 milhões de dólares, 1.264 falsos negativos e recall de 79,62%. O lucro por registro tem média externa de 328,81 dólares e desvio-padrão de 51,21.

O champion supera o challenger em 203,05 mil dólares no lucro OOF agregado e registra 23 falsos negativos a menos. Ao mesmo tempo, a alternância observada nos folds externos justifica manter o LightGBM como alternativa para comparação, contingência e monitoramento.

Os detalhes estão disponíveis em `metricas_folds_ncv.csv`, `selecao_folds_ncv.csv` e `frequencia_selecao_ncv.csv`.

### 3.7 Importâncias das variáveis

O gráfico apresenta as dez variáveis mais importantes do XGBoost balanceado reajustado em toda a base. Destacam-se o percentual da renda comprometida com o empréstimo, a posse de imóvel próprio, a classificação de crédito C, a existência de hipoteca, a finalidade de empreendimento e a taxa de juros.

As importâncias são calculadas de forma diferente entre algoritmos, devem ser comparadas dentro de cada modelo e não indicam direção do efeito nem causalidade.

![Importâncias do XGBoost balanceado no ajuste final](outputs/figures/importancias_xgboost_balanceado_top10_ncv.png)

### 3.8 Interpretabilidade do modelo

O SHAP aprofunda a análise ao apresentar a direção e a intensidade da contribuição das variáveis para as previsões do XGBoost balanceado reajustado. Cada ponto representa um registro. A posição horizontal mostra a contribuição da variável: valores SHAP positivos aumentam a saída do modelo associada ao risco de inadimplência e valores negativos a reduzem. Quanto mais distante de zero, maior o impacto daquela variável na previsão. As variáveis são ordenadas pelo impacto absoluto médio.

As cores representam o valor observado da variável: rosa indica valores mais altos e azul valores mais baixos. Em variáveis binárias, rosa representa a presença e azul a ausência da característica. Os valores SHAP são contribuições para a saída do modelo e não devem ser interpretados diretamente como variações em pontos percentuais da probabilidade.

Os principais padrões indicam que taxas de juros e percentuais da renda comprometidos com o empréstimo mais altos elevam o risco previsto, enquanto rendas anuais mais altas o reduzem. Valores maiores de empréstimo e a classificação de crédito D também tendem a aumentar o risco. Imóvel próprio, hipoteca e empréstimos destinados a empreendimento ou educação aparecem associados à redução do risco previsto. Esses efeitos podem refletir relações não lineares e interações aprendidas pelo XGBoost.

Como o gráfico utiliza o modelo reajustado e toda a base transformada, ele é um artefato de interpretação do modelo final, e não uma nova avaliação OOF. Os padrões descrevem associações utilizadas pelo modelo e não comprovam relações causais.

![SHAP do XGBoost balanceado no ajuste final](outputs/figures/shap_xgboost_balanceado_ncv.png)

## 4. Considerações

O nested cross-validation separa a seleção interna da avaliação externa e reduz a dependência de uma única divisão aleatória. Os resultados confirmam a proximidade entre XGBoost e LightGBM balanceados e fornecem medidas de dispersão e frequência de seleção que não estavam disponíveis em um holdout simples.

Ainda não existe um modelo superior em todos os critérios. O XGBoost balanceado foi escolhido como champion por combinar resultado financeiro, estabilidade e identificação de inadimplentes. O XGBoost original lidera F1-score, ROC AUC e PR-AUC, enquanto o LightGBM original apresenta a maior precisão.

Entre os próximos passos estão:

- incorporar prazo, recuperação, custo de capital e despesas à avaliação financeira;
- otimizar o limiar de classificação dentro dos folds internos;
- avaliar calibração das probabilidades, explicabilidade e desempenho por subgrupos;
- testar os modelos em dados externos e monitorar estabilidade e degradação ao longo do tempo.

As principais limitações são o uso de um dataset público sem dimensão temporal nem validação externa, a exclusão completa dos registros com valores ausentes e o limiar fixo de 0,5. O nested CV reduz o viés de seleção interno, mas não mede mudança temporal ou representatividade em outra carteira. O lucro continua sendo uma aproximação de um período com perda integral em caso de inadimplência e sem outros custos ou recuperações. As importâncias e valores SHAP descrevem associações do modelo e não representam efeitos causais.
