import ast
import importlib
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

rm = importlib.import_module("5_roda_modelos")
aj = importlib.import_module("6_ajusta_modelos")

TABELAS = rm.config.RESULTADOS / "tabelas"
REPETICOES = 10
INFORMACOES = {**{c: [c] for c in rm.CATEGORICAS + ["idade", "feriado"]},
               "dia_semana": ["dia_semana_seno", "dia_semana_cosseno"],
               "dia_mes": ["dia_mes_seno", "dia_mes_cosseno"]}
GRUPOS = {"especialidade": "ESPEC", "hospital": "CNES"}
FAIXAS = [(0, 3, "até 3 dias"), (4, 14, "4 a 14 dias"), (15, np.inf, "15 ou mais")]


def melhores_da_busca():
    busca = pd.read_csv(aj.REGISTRO, sep=";", decimal=",")
    melhores = busca.loc[busca.groupby(["modelo", "treino"]).mae.idxmin()]
    return {(l.modelo, l.treino == "log"): (ast.literal_eval(l.parametros),
                                             int(l.parada) if pd.notna(l.parada) else None)
            for l in melhores.itertuples()}


def treina_final(nome, parametros, parada, x, y):
    parametros = dict(parametros)
    if nome == "regressao_linear":
        return clone(rm.MODELOS[nome]).fit(x, y)
    if nome == "random_forest":
        return RandomForestRegressor(random_state=rm.SEMENTE, n_jobs=-1, **parametros).fit(x, y)
    if nome == "xgboost":
        return XGBRegressor(n_estimators=parada, random_state=rm.SEMENTE, **parametros).fit(x, y)
    classe = aj.MLPComReLU if parametros.pop("relu_na_saida", False) else MLPRegressor
    modelo = classe(random_state=np.random.RandomState(rm.SEMENTE), **parametros)
    for _ in range(parada):
        modelo.partial_fit(x, y)
    return modelo


def metricas(real, bruto):
    previsto = aj.troca_negativos(bruto)
    return {"mae": round(mean_absolute_error(real, previsto), 4),
            "rmse": round(root_mean_squared_error(real, previsto), 4),
            "r2": round(r2_score(real, previsto), 4),
            "mae_sem_troca": round(mean_absolute_error(real, bruto), 4),
            "rmse_sem_troca": round(root_mean_squared_error(real, bruto), 4),
            "r2_sem_troca": round(r2_score(real, bruto), 4),
            "negativos": int((bruto < 0).sum()),
            "soma_prevista_sobre_real": round(previsto.sum() / real.sum(), 4)}


def erro_por_grupo(chave, avaliado, real, previsto):
    linhas = []
    for grupo, coluna in GRUPOS.items():
        for valor, indices in avaliado.groupby(coluna).indices.items():
            linhas.append({**chave, "grupo": grupo, "valor": valor, "n": len(indices),
                           "mae": round(mean_absolute_error(real[indices], previsto[indices]), 4),
                           "vies": round((previsto[indices] - real[indices]).mean(), 4)})
    for inicio, fim, rotulo in FAIXAS:
        dentro = (real >= inicio) & (real <= fim)
        linhas.append({**chave, "grupo": "faixa", "valor": rotulo, "n": int(dentro.sum()),
                       "mae": round(mean_absolute_error(real[dentro], previsto[dentro]), 4),
                       "vies": round((previsto[dentro] - real[dentro]).mean(), 4)})
    return linhas


def importancia(chave, preve, x_avaliado, real, embaralhamentos):
    base = mean_absolute_error(real, aj.troca_negativos(preve(x_avaliado)))
    linhas = []
    for informacao, colunas in INFORMACOES.items():
        aumentos = []
        for ordem in embaralhamentos[informacao]:
            embaralhado = x_avaliado.copy()
            embaralhado[colunas] = x_avaliado[colunas].to_numpy()[ordem]
            aumentos.append(mean_absolute_error(real, aj.troca_negativos(preve(embaralhado))) - base)
        linhas.append({**chave, "informacao": informacao, "aumento_mae": round(np.mean(aumentos), 4),
                       "desvio": round(np.std(aumentos), 4), "repeticoes": len(aumentos)})
    return linhas


def coloca_literatura(arquivo):
    linhas = arquivo.read_text(encoding="utf-8").splitlines()
    valores = {**rm.LITERATURA, "modelo": "literatura"}
    referencia = ";".join(valores.get(coluna, "") for coluna in linhas[0].split(";"))
    arquivo.write_text("\n".join([linhas[0], referencia, *linhas[1:]]) + "\n", encoding="utf-8")


def main(conjunto):
    treino = rm.le("treino")
    if conjunto == "teste":
        treino = pd.concat([treino, rm.le("validacao")], ignore_index=True)
    avaliado = rm.le(conjunto).reset_index(drop=True)
    grupos = rm.aprende_grupos(treino)
    x_treino, x_avaliado = rm.prepara(treino, grupos), rm.prepara(avaliado, grupos)
    real = avaliado[rm.ALVO].to_numpy()
    sorteio = np.random.RandomState(rm.SEMENTE)
    embaralhamentos = {i: [sorteio.permutation(len(avaliado)) for _ in range(REPETICOES)] for i in INFORMACOES}
    melhores = melhores_da_busca()
    previsoes = avaliado[["N_AIH", "ESPEC", "CNES", rm.ALVO]].copy()
    resultados, grupos_erro, importancias = [], [], []
    for nome in rm.MODELOS:
        codificador = rm.cria_codificador(nome, x_treino)
        x = codificador.transform(x_treino)
        for log in (False, True):
            parametros, parada = melhores.get((nome, log), ({}, None))
            y = np.log1p(treino[rm.ALVO]) if log else treino[rm.ALVO]
            modelo = treina_final(nome, parametros, parada, x, y)

            def preve(xa):
                return aj.em_dias(modelo.predict(codificador.transform(xa)), log)

            chave = {"modelo": nome, "treino": "log" if log else "dias"}
            bruto = preve(x_avaliado)
            previsto = aj.troca_negativos(bruto)
            resultados.append({**chave, "parametros": str(parametros), "parada": parada,
                               "n": len(real), **metricas(real, bruto)})
            previsoes[f"{nome}_{chave['treino']}"] = previsto.round(4)
            grupos_erro += erro_por_grupo(chave, avaliado, real, previsto)
            importancias += importancia(chave, preve, x_avaliado, real, embaralhamentos)
            r = resultados[-1]
            print(f"{nome} ({chave['treino']}): MAE {r['mae']:.4f}  RMSE {r['rmse']:.4f}  R2 {r['r2']:.4f}  "
                  f"MAE sem troca {r['mae_sem_troca']:.4f}  negativos {r['negativos']}", flush=True)
    TABELAS.mkdir(parents=True, exist_ok=True)
    for nome, tabela in [("avaliacao", resultados), ("erro_por_grupo", grupos_erro),
                         ("importancia", importancias), ("previsoes", previsoes)]:
        pd.DataFrame(tabela).to_csv(TABELAS / f"{nome}_{conjunto}.csv", index=False, encoding="utf-8",
                                    sep=";", decimal=",")
    coloca_literatura(TABELAS / f"avaliacao_{conjunto}.csv")


if __name__ == "__main__":
    main("teste" if "teste" in sys.argv else "validacao")
