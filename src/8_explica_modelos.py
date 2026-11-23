import importlib

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

rm = importlib.import_module("5_roda_modelos")
aj = importlib.import_module("6_ajusta_modelos")
av = importlib.import_module("7_avalia_modelos")

TABELAS = av.TABELAS
INFORMACOES = list(av.INFORMACOES)
INTERNACOES = 1000
FUNDO = 100
AVALIACOES_SHAP = 1000
AMOSTRAS_LIME = 5000
LOTE = 20000


def colunas_por_informacao(codificador):
    nomes = list(codificador.get_feature_names_out())
    posicoes = {}
    for informacao, originais in av.INFORMACOES.items():
        if informacao in rm.CATEGORICAS:
            posicoes[informacao] = [i for i, n in enumerate(nomes) if n.startswith(f"categoricas__{informacao}_")]
        else:
            posicoes[informacao] = [i for i, n in enumerate(nomes) if n.split("__")[1] in originais]
    return posicoes


class Tradutor:
    def __init__(self, matriz, posicoes):
        self.total = matriz.shape[1]
        self.posicoes = [posicoes[i] for i in INFORMACOES]
        self.idade = INFORMACOES.index("idade")
        self.tabelas = []
        self.codigos = np.empty((len(matriz), len(INFORMACOES)))
        for j, colunas in enumerate(self.posicoes):
            if j == self.idade:
                self.tabelas.append(None)
                self.codigos[:, j] = matriz[:, colunas[0]]
            else:
                tabela, codigo = np.unique(matriz[:, colunas], axis=0, return_inverse=True)
                self.tabelas.append(tabela)
                self.codigos[:, j] = codigo.ravel()

    def matriz(self, codigos):
        saida = np.empty((len(codigos), self.total))
        for j, colunas in enumerate(self.posicoes):
            if j == self.idade:
                saida[:, colunas[0]] = codigos[:, j]
            else:
                saida[:, colunas] = self.tabelas[j][codigos[:, j].astype(int)]
        return saida


def preve_em_lotes(modelo, tradutor, log, codigos):
    partes = [modelo.predict(tradutor.matriz(codigos[i:i + LOTE])) for i in range(0, len(codigos), LOTE)]
    return aj.troca_negativos(aj.em_dias(np.concatenate(partes), log))


def estrutura(nome, modelo, x, posicoes):
    if nome == "regressao_linear":
        contribuicao = {i: (x[:, c] @ modelo.coef_[c]).std() for i, c in posicoes.items()}
    elif nome == "random_forest":
        contribuicao = {i: modelo.feature_importances_[c].sum() for i, c in posicoes.items()}
    elif nome == "xgboost":
        ganho = modelo.get_booster().get_score(importance_type="total_gain")
        contribuicao = {i: sum(ganho.get(f"f{k}", 0.0) for k in c) for i, c in posicoes.items()}
    else:
        return None
    return contribuicao


def calcula_shap(funcao, tradutor, fundo, amostra, semente):
    mascara = shap.maskers.Independent(tradutor.codigos[fundo], max_samples=len(fundo))
    explicador = shap.explainers.Permutation(funcao, mascara, seed=semente)
    resultado = explicador(tradutor.codigos[amostra], max_evals=AVALIACOES_SHAP, batch_size=200, silent=True)
    return resultado.values, resultado.base_values


def calcula_lime(funcao, tradutor, n_treino, amostra, semente):
    categoricas = [j for j in range(len(INFORMACOES)) if j != tradutor.idade]
    explicador = LimeTabularExplainer(tradutor.codigos[:n_treino], mode="regression", feature_names=INFORMACOES,
                                      categorical_features=categoricas, discretize_continuous=True,
                                      random_state=semente)
    pesos, qualidade = np.zeros((len(amostra), len(INFORMACOES))), []
    for linha, posicao in enumerate(amostra):
        explicacao = explicador.explain_instance(tradutor.codigos[posicao], funcao, num_features=len(INFORMACOES),
                                                 num_samples=AMOSTRAS_LIME)
        for j, peso in explicacao.as_map()[1]:
            pesos[linha, j] = peso
        qualidade.append(explicacao.score)
    return pesos, np.array(qualidade)


def linhas_do_metodo(chave, metodo, valores):
    positivos = sum(max(v, 0) for v in valores.values())
    return [{**chave, "metodo": metodo, "informacao": i, "valor": round(v, 6),
             "percentual": round(100 * max(v, 0) / positivos, 2)} for i, v in valores.items()]


def main(semente=rm.SEMENTE, sufixo=""):
    treino = pd.concat([rm.le("treino"), rm.le("validacao")], ignore_index=True)
    teste = rm.le("teste").reset_index(drop=True)
    grupos = rm.aprende_grupos(treino)
    x_treino, x_teste = rm.prepara(treino, grupos), rm.prepara(teste, grupos)
    conferencia = pd.read_csv(TABELAS / "previsoes_teste.csv", sep=";", decimal=",")
    permutacao = pd.read_csv(TABELAS / "importancia_teste.csv", sep=";", decimal=",")
    sorteio = np.random.RandomState(semente)
    fundo_treino = sorteio.choice(len(treino), FUNDO, replace=False)
    amostra_teste = sorteio.choice(len(teste), INTERNACOES, replace=False)
    melhores = av.melhores_da_busca()
    resumo, por_internacao = [], []
    for nome in rm.MODELOS:
        codificador = rm.cria_codificador(nome, x_treino)
        matriz_treino = codificador.transform(x_treino)
        matriz_teste = codificador.transform(x_teste)
        posicoes = colunas_por_informacao(codificador)
        tradutor = Tradutor(np.vstack([matriz_treino, matriz_teste[amostra_teste]]), posicoes)
        amostra = np.arange(len(treino), len(treino) + INTERNACOES)
        for log in (False, True):
            forma = "log" if log else "dias"
            chave = {"modelo": nome, "treino": forma}
            parametros, parada = melhores.get((nome, log), ({}, None))
            alvo = np.log1p(treino[rm.ALVO]) if log else treino[rm.ALVO]
            modelo = av.treina_final(nome, parametros, parada, matriz_treino, alvo)
            previsto = aj.troca_negativos(aj.em_dias(modelo.predict(matriz_teste), log))
            diferenca = np.abs(previsto.round(4) - conferencia[f"{nome}_{forma}"].to_numpy()).max()
            if diferenca > 1e-3:
                raise ValueError(f"{nome} {forma}: previsao diferente da avaliacao, {diferenca}")

            def funcao(codigos):
                return preve_em_lotes(modelo, tradutor, log, codigos)

            valores_shap, base = calcula_shap(funcao, tradutor, fundo_treino, amostra, semente)
            soma = base + valores_shap.sum(axis=1)
            if np.abs(soma - previsto[amostra_teste]).max() > 1e-3:
                raise ValueError(f"{nome} {forma}: valores SHAP nao somam a previsao")
            pesos_lime, qualidade = calcula_lime(funcao, tradutor, len(treino), amostra, semente)
            dados_permutacao = permutacao[(permutacao.modelo == nome) & (permutacao.treino == forma)]
            resumo += linhas_do_metodo(chave, "permutacao", dict(zip(dados_permutacao.informacao,
                                                                     dados_permutacao.aumento_mae)))
            pela_estrutura = estrutura(nome, modelo, matriz_treino, posicoes)
            if pela_estrutura:
                resumo += linhas_do_metodo(chave, "estrutura", pela_estrutura)
            resumo += linhas_do_metodo(chave, "shap", dict(zip(INFORMACOES, np.abs(valores_shap).mean(axis=0))))
            resumo += linhas_do_metodo(chave, "lime", dict(zip(INFORMACOES, np.abs(pesos_lime).mean(axis=0))))
            for metodo, valores in (("shap", valores_shap), ("lime", pesos_lime)):
                tabela = pd.DataFrame(np.round(valores, 5), columns=INFORMACOES)
                tabela.insert(0, "N_AIH", teste.N_AIH.to_numpy()[amostra_teste])
                tabela.insert(1, "modelo", nome)
                tabela.insert(2, "treino", forma)
                tabela.insert(3, "metodo", metodo)
                tabela.insert(4, "previsto", np.round(previsto[amostra_teste], 5))
                tabela.insert(5, "base", np.round(base, 5) if metodo == "shap" else np.nan)
                tabela.insert(6, "qualidade", np.nan if metodo == "shap" else np.round(qualidade, 4))
                por_internacao.append(tabela)
            print(f"{nome} ({forma}): SHAP soma ok, LIME qualidade media {qualidade.mean():.3f}", flush=True)
    pd.DataFrame(resumo).to_csv(TABELAS / f"explicacao_teste{sufixo}.csv", index=False, encoding="utf-8",
                                sep=";", decimal=",")
    pd.concat(por_internacao).to_csv(TABELAS / f"explicacao_amostra_teste{sufixo}.csv", index=False, encoding="utf-8",
                                     sep=";", decimal=",")


if __name__ == "__main__":
    main()
