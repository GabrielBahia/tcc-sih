import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import config

SPLITS = config.SPLITS
REGISTRO = config.RESULTADOS / "tabelas" / "experimentos.csv"
LITERATURA = {"experimento": "literatura", "mae": "1,4 a 3", "rmse": "2,1 a 3,8", "r2": "0,15 a 0,45"}

SEMENTE = 42
ALVO = "los"
CATEGORICAS = ["SEXO", "CAR_INT", "RACA_COR", "NAT_JUR", "ESPEC", "CNES", "cid_item",
               "MUNIC_RES", "PROC_SOLIC"]
DATAS = ["dia_semana_seno", "dia_semana_cosseno", "dia_mes_seno", "dia_mes_cosseno", "feriado"]
FERIADOS = pd.to_datetime([
    "2024-01-01", "2024-03-29", "2024-04-21", "2024-05-01", "2024-05-30", "2024-06-13",
    "2024-09-07", "2024-10-12", "2024-11-02", "2024-11-15", "2024-11-20", "2024-12-25",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-06-13", "2025-06-19",
    "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-11-20", "2025-12-25"])

MODELOS = {
    "regressao_linear": LinearRegression(),
    "random_forest": RandomForestRegressor(random_state=SEMENTE, n_jobs=-1),
    "xgboost": XGBRegressor(random_state=SEMENTE),
    "mlp": MLPRegressor(random_state=SEMENTE),
}


def le(conjunto):
    return pd.read_parquet(SPLITS / f"{conjunto}.parquet")


def mais_frequentes(valores):
    contagem = valores.value_counts()
    posicao = (contagem.cumsum() < 0.95 * len(valores)).sum()
    return set(contagem.index[contagem >= contagem.iloc[posicao]])


def aprende_grupos(treino):
    procedimentos = mais_frequentes(treino.PROC_SOLIC)
    raros = treino.PROC_SOLIC[~treino.PROC_SOLIC.isin(procedimentos)]
    formas = set(raros.str[:6])
    municipios = mais_frequentes(treino.MUNIC_RES)
    return procedimentos, formas, municipios


def datas(d):
    entrada = pd.to_datetime(d.DT_INTER.astype(str), format="%Y%m%d")
    if not entrada.dt.year.isin([2024, 2025]).all():
        raise ValueError("data de entrada fora de 2024 e 2025, sem calendario de feriados")
    semana = 2 * np.pi * entrada.dt.dayofweek / 7
    dia = 2 * np.pi * entrada.dt.day / 31
    return pd.DataFrame({"dia_semana_seno": np.sin(semana), "dia_semana_cosseno": np.cos(semana),
                         "dia_mes_seno": np.sin(dia), "dia_mes_cosseno": np.cos(dia),
                         "feriado": entrada.isin(FERIADOS).astype(int)}, index=d.index)


def prepara(d, grupos):
    procedimentos, formas, municipios = grupos
    x = d[CATEGORICAS + ["idade"]].copy()
    forma = d.PROC_SOLIC.str[:6]
    x["PROC_SOLIC"] = d.PROC_SOLIC.where(d.PROC_SOLIC.isin(procedimentos),
                                         forma.where(forma.isin(formas), "desconhecido"))
    x["MUNIC_RES"] = d.MUNIC_RES.where(d.MUNIC_RES.isin(municipios), "outros")
    return x.join(datas(d))


def cria_codificador(nome, x_treino):
    if nome == "regressao_linear":
        retira = [x_treino[c].value_counts().index[0] for c in CATEGORICAS]
    else:
        retira = "if_binary"
    idade = StandardScaler() if nome == "mlp" else "passthrough"
    codificador = ColumnTransformer(
        [("categoricas", OneHotEncoder(drop=retira, handle_unknown="ignore"), CATEGORICAS),
         ("numericas", idade, ["idade"]),
         ("datas", "passthrough", DATAS)],
        sparse_threshold=0)
    return codificador.fit(x_treino)


def registra(nome, n, mae, rmse, r2, observacao):
    linha = pd.DataFrame([{
        "experimento": "configuracao_padrao", "modelo": nome,
        "conjunto": "validacao", "preditoras": 13, "n": n, "mae": round(mae, 4),
        "rmse": round(rmse, 4), "r2": round(r2, 4), "semente": SEMENTE, "observacao": observacao}])
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRO.exists():
        pd.DataFrame([LITERATURA], columns=linha.columns).to_csv(REGISTRO, index=False, encoding="utf-8",
                                                                 sep=";")
    linha.to_csv(REGISTRO, mode="a", header=False, index=False, encoding="utf-8", sep=";", decimal=",")


def main(nomes, log=False):
    treino, validacao = le("treino"), le("validacao")
    grupos = aprende_grupos(treino)
    x_treino, x_validacao = prepara(treino, grupos), prepara(validacao, grupos)
    alvo = np.log1p(treino[ALVO]) if log else treino[ALVO]
    for nome in nomes:
        codificador = cria_codificador(nome, x_treino)
        modelo = MODELOS[nome]
        modelo.fit(codificador.transform(x_treino), alvo)
        previsto = modelo.predict(codificador.transform(x_validacao))
        if log:
            previsto = np.expm1(previsto)
        real = validacao[ALVO]
        mae = mean_absolute_error(real, previsto)
        rmse = root_mean_squared_error(real, previsto)
        r2 = r2_score(real, previsto)
        registra(nome, len(real), mae, rmse, r2, "alvo em log" if log else "")
        print(f"{nome}{' (alvo em log)' if log else ''}: MAE {mae:.4f}  RMSE {rmse:.4f}  R2 {r2:.4f}")


if __name__ == "__main__":
    log = "log" in sys.argv
    main([a for a in sys.argv[1:] if a != "log"] or list(MODELOS), log)
