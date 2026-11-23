# TCC: previsão do tempo de internação (SIH/SUS, Juiz de Fora)

Código do TCC de Gabriel Gomes Bahia (UFJF, Ciência da Computação), orientado pelo Prof. Heder Soares Bernardino.

O projeto prevê quantos dias uma internação vai durar, usando só o que se sabe na entrada do paciente, a partir dos dados públicos de internação do SUS (SIH/SUS) nos hospitais de Juiz de Fora.

## Como rodar

```bash
pip install -r requirements.txt
python src/1_baixa_dados.py
python src/2_junta_meses.py
python src/3_trata_base.py
python src/4_divide.py
python src/testa_modelos.py
python src/5_roda_modelos.py
python src/6_ajusta_modelos.py
python src/6_ajusta_modelos.py log
python src/7_avalia_modelos.py
python src/7_avalia_modelos.py teste
python src/8_explica_modelos.py
```

Usa Python 3.10, com os comandos rodados a partir da raiz do repositório.

## Rodar um modelo só

```bash
python src/5_roda_modelos.py regressao_linear
python src/5_roda_modelos.py regressao_linear log
```

- Os nomes aceitos são `regressao_linear`, `random_forest`, `xgboost` e `mlp`. Dá para passar mais de um, e sem nome rodam os quatro.
- Com `log` no fim, o modelo aprende com log(1 + dias) e a previsão volta para dias antes de medir.
- Precisa de `dados/04_splits/treino.parquet` e `validacao.parquet`, gravados pelo `4_divide.py`.
- Mostra MAE, RMSE e R² na tela e acrescenta uma linha em `resultados/tabelas/experimentos.csv` a cada rodada.
- A primeira linha do `experimentos.csv` traz a faixa de MAE, RMSE e R² da revisão da literatura, para comparar.
- O aviso `Found unknown categories in columns [6, 8]` é esperado: são CID e procedimentos da validação que não aparecem no treino, e viram zero em todas as colunas.

## Ajustar os parâmetros

```bash
python src/6_ajusta_modelos.py
python src/6_ajusta_modelos.py log
```

- Testa na validação todas as combinações da grade de cada modelo: 24 no Random Forest, 24 no XGBoost e 64 na MLP. A regressão linear não tem o que ajustar.
- Na MLP, `relu_na_saida` testa a rede com e sem a ReLU na saída. Com ela, a saída é o maior valor entre 0 e a conta da última camada, e a rede nunca prevê menos de 0. O scikit-learn não tem essa opção, então a classe `MLPComReLU` troca a saída da `MLPRegressor` por ReLU e zera a correção da última camada nas internações em que a saída deu 0.
- Cada combinação treina no treino e é medida pelo MAE em dias na validação. O teste não é usado.
- O XGBoost e a MLP param de crescer quando o MAE da validação passa 20 árvores ou 20 épocas sem melhorar, e ficam com o ponto de menor MAE.
- As previsões negativas são trocadas por zero antes de medir, tanto na escolha da combinação quanto na parada do XGBoost e da MLP.
- Grava uma linha por combinação em `resultados/tabelas/busca_em_grade.csv`, assim que ela termina, com MAE, RMSE e R² medidos com a troca, o MAE sem a troca (`mae_sem_troca`), a quantidade de previsões negativas e o tempo.
- Dá para passar só alguns modelos, como em `python src/6_ajusta_modelos.py mlp log`. As duas escalas levam cerca de 1 hora e 40 minutos, a maior parte na MLP.

## Avaliar os modelos escolhidos

```bash
python src/7_avalia_modelos.py
python src/7_avalia_modelos.py teste
```

- Pega no `busca_em_grade.csv` a combinação de menor MAE de cada modelo em cada forma de treino. A regressão linear roda com o padrão.
- Sem argumento, treina no treino e mede na validação, para conferir o script sem usar o teste. Os resultados repetem os da busca em grade.
- Com `teste`, treina de novo com o treino e a validação juntos, refazendo os cortes, a codificação e a padronização da idade, e mede no teste. O XGBoost e a MLP usam o número de árvores e de épocas encontrado na validação.
- Mede MAE, RMSE e R² com e sem a troca das previsões negativas por zero, quantas foram trocadas e a soma das previsões sobre a soma real.
- Calcula o MAE e o viés (média de previsto menos real) por especialidade, por hospital e por faixa do tempo real: até 3 dias, de 4 a 14 e 15 ou mais.
- Mede a importância de cada uma das 13 informações por permutação: embaralha a informação entre as internações, 10 vezes, e guarda o aumento médio do MAE. O seno e o cosseno do mesmo dia são embaralhados juntos.
- Grava, com `_validacao` ou `_teste` no nome, em `resultados/tabelas`: `avaliacao`, `erro_por_grupo`, `importancia` e `previsoes` (uma linha por internação). Cada rodada substitui os arquivos da rodada anterior do mesmo conjunto.

## Painel

```bash
streamlit run src/painel.py
```

- Abre em `http://localhost:8501`, só na própria máquina. O envio de estatísticas do Streamlit fica desligado em `.streamlit/config.toml`.
- A aba Variáveis mostra as preditoras, o que cada uma é, um resumo no treino e quantas colunas viram em cada modelo.
- A aba Modelo escolhe o modelo, o treino em dias ou em log e a configuração. A padrão roda como o `5_roda_modelos.py`. A de escolher parâmetros roda como o `6_ajusta_modelos.py` e começa com a melhor combinação da busca em grade. Mais de um valor num parâmetro testa todas as combinações.
- Durante o treino mostra o log e a curva do MAE na validação por árvore ou por época. Depois mostra as métricas, os erros por duração real, uma árvore ou as camadas da MLP, e baixa o resultado, as previsões e o log.
- A aba Histórico mostra o `experimentos.csv`, o `busca_em_grade.csv` e as rodadas do painel.
- Cada rodada grava uma linha em `resultados/tabelas/painel.csv`. O teste não é carregado.

## O que cada script faz

| script | o que faz | grava |
|---|---|---|
| `config.py` | guarda as escolhas fixas: município, período, datas da divisão e pastas | nada |
| `1_baixa_dados.py` | baixa do DATASUS os arquivos mensais de Minas Gerais de 2008 a 2026 e guarda só as internações em hospitais de Juiz de Fora | `dados/00_dbc`, `dados/01_competencias`, `logs/extracao_datasus.csv` |
| `2_junta_meses.py` | junta os meses numa base só e separa as internações que começaram em 2024 e 2025 | `dados/02_consolidado` |
| `3_trata_base.py` | junta as AIH da mesma internação e retira as que cruzam a borda do período, as sem desfecho, dois hospitais e os óbitos, calcula a idade e agrupa o CID | `dados/03_tratado/sih_jf_tratado.parquet`, `resultados/tabelas/tratamento_passo_a_passo.csv` |
| `4_divide.py` | divide pela data de entrada: treino em 2024, validação no 1º semestre de 2025 e teste no 2º semestre | `dados/04_splits` |
| `5_roda_modelos.py` | prepara as variáveis, treina regressão linear, Random Forest, XGBoost e MLP e mede MAE, RMSE e R² na validação | `resultados/tabelas/experimentos.csv` |
| `6_ajusta_modelos.py` | busca em grade dos parâmetros do Random Forest, do XGBoost e da MLP, medida na validação | `resultados/tabelas/busca_em_grade.csv` |
| `7_avalia_modelos.py` | treina de novo a melhor combinação de cada modelo e mede no teste, com e sem a troca dos negativos, o erro por grupo e a importância das informações | `resultados/tabelas/avaliacao_*.csv`, `erro_por_grupo_*.csv`, `importancia_*.csv`, `previsoes_*.csv` |
| `8_explica_modelos.py` | treina de novo os 8 modelos, confere as previsões do teste e mede a importância das 13 informações pela estrutura do modelo, por SHAP e por LIME, em 1.000 internações do teste | `resultados/tabelas/explicacao_teste.csv` e `explicacao_amostra_teste.csv` |
| `testa_modelos.py` | confere a preparação na base real, os 4 modelos numa tabela inventada e a correção da ReLU na saída da MLP contra a conta feita pela fórmula | nada |
| `painel.py` | página local para ver as variáveis, rodar um modelo com parâmetros escolhidos e ver o resultado | `resultados/tabelas/painel.csv` |

O primeiro script leva de 1 a 1,5 hora na primeira vez e pode ser rodado de novo se a conexão cair: mês já baixado não é baixado outra vez.

## Dados

```
dados/
  00_dbc/            arquivos do DATASUS como foram baixados
  01_competencias/   um arquivo por mês, só Juiz de Fora
  02_consolidado/    todos os meses juntos e o período de 2024 e 2025
  03_tratado/        a base depois das regras
  04_splits/         treino, validação e teste
  05_auxiliar/       a Lista de Tabulação para Morbidade da CID-10, usada para agrupar o diagnóstico
```

Tudo em `dados`, fora a pasta `05_auxiliar`, é gerado pelos scripts e pode ser refeito do zero.

## Fonte dos dados

Arquivos de AIH reduzida do SIH/SUS, do repositório público do DATASUS em `ftp.datasus.gov.br`. O recorte é pelo município do hospital (`MUNIC_MOV = 313670`), e não pelo de residência do paciente.
