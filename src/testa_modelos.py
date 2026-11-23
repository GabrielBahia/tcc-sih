import importlib
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

rm = importlib.import_module("5_roda_modelos")
aj = importlib.import_module("6_ajusta_modelos")

warnings.filterwarnings("ignore")
placar = {"ok": 0, "falhas": 0}


def confere(descricao, ok):
    placar["ok" if ok else "falhas"] += 1
    if not ok:
        print(f"FALHA  {descricao}")


def frequentes_por_outro_caminho(valores):
    contagem = Counter(valores).most_common()
    acumulado = 0
    for _, n in contagem:
        acumulado += n
        if acumulado >= 0.95 * len(valores):
            return {valor for valor, m in contagem if m >= n}


def tabela_inventada(n, semente):
    r = np.random.RandomState(semente)
    d = pd.DataFrame({
        "SEXO": r.choice(["1", "3"], n),
        "CAR_INT": r.choice(["01", "02"], n, p=[.3, .7]),
        "RACA_COR": r.choice(["01", "03", "04"], n, p=[.3, .5, .2]),
        "NAT_JUR": r.choice(["1023", "3999"], n),
        "ESPEC": r.choice(["01", "02", "03"], n),
        "CNES": r.choice(["1111111", "2222222"], n),
        "cid_item": r.choice(["A", "B", "C"], n),
        "MUNIC_RES": r.choice(["313670", "313680", "310000"], n, p=[.8, .17, .03]),
        "PROC_SOLIC": r.choice(["0303140151", "0303140160", "0411010034", "0411010042"], n,
                               p=[.5, .3, .18, .02]),
        "idade": r.uniform(0, 90, n).round(1),
        "DT_INTER": (pd.Timestamp("2024-01-01")
                     + pd.to_timedelta(r.randint(0, 366, n), unit="D")).strftime("%Y%m%d")})
    entrada = pd.to_datetime(d.DT_INTER, format="%Y%m%d")
    d["los"] = (2 + 0.1 * d.idade + 3 * (d.CAR_INT == "01") + 1.5 * (d.RACA_COR == "04")
                + np.sin(2 * np.pi * entrada.dt.dayofweek / 7) + 2 * entrada.isin(rm.FERIADOS))
    return d


def regras_inventadas():
    empate = pd.Series(["A"] * 31 + ["B", "C", "D", "E", "F"] * 2)
    confere("empate no limite dos 95%: todos os empatados entram",
            rm.mais_frequentes(empate) == set("ABCDEF"))
    confere("95% exatos nao puxam valor a mais",
            rm.mais_frequentes(pd.Series(["A"] * 19 + ["B"])) == {"A"})
    d = pd.DataFrame({"DT_INTER": ["20240613", "20240531", "20250619", "20241231"]})
    x = rm.datas(d)
    confere("feriado marcado so em 13/06/2024 e 19/06/2025", list(x.feriado) == [1, 0, 1, 0])
    confere("quinta-feira vira angulo 3 de 7",
            abs(x.dia_semana_seno[0] - np.sin(2 * np.pi * 3 / 7)) < 1e-12)
    confere("dia 31 fecha o ciclo do mes", abs(x.dia_mes_cosseno[3] - 1) < 1e-12)
    try:
        rm.datas(pd.DataFrame({"DT_INTER": ["20230101"]}))
        confere("data fora de 2024 e 2025 gera erro", False)
    except ValueError:
        confere("data fora de 2024 e 2025 gera erro", True)


def relu_na_saida():
    r = np.random.RandomState(0)
    x, y = r.rand(40, 6), r.rand(40) * 10
    for camadas in [(8,), (8, 5)]:
        mlp = aj.MLPComReLU(hidden_layer_sizes=camadas, alpha=0.01, solver="lbfgs", max_iter=1,
                            random_state=1).fit(x, y)
        z = x
        for i in range(mlp.n_layers_ - 1):
            z = z @ mlp.coefs_[i] + mlp.intercepts_[i]
            z = np.maximum(z, 0) if i < mlp.n_layers_ - 2 else z
        mlp.intercepts_[-1] -= np.median(z)
        unidades = [x.shape[1], *camadas, 1]
        listas = [[x], [], [], []]
        for a, b in zip(unidades, unidades[1:]):
            listas[0].append(np.empty((len(x), b)))
            listas[1].append(np.empty((len(x), b)))
            listas[2].append(np.empty((a, b)))
            listas[3].append(np.empty(b))
        theta = np.hstack([p.ravel() for p in mlp.coefs_ + mlp.intercepts_])
        perda = lambda t: mlp._loss_grad_lbfgs(t, x, y.reshape(-1, 1), None, *listas)
        calculado = perda(theta)[1].copy()
        passo = 1e-6 * np.eye(theta.size)
        numerico = np.array([(perda(theta + d)[0] - perda(theta - d)[0]) / 2e-6 for d in passo])
        mlp._unpack(theta)
        previsto = mlp.predict(x)
        confere(f"relu na saida {camadas}: gradiente igual ao calculado pela formula",
                np.abs(calculado - numerico).max() < 1e-6)
        confere(f"relu na saida {camadas}: metade das previsoes em zero, nenhuma negativa",
                previsto.min() == 0 and (previsto == 0).sum() == 20)
    confere("grade da mlp com e sem relu na saida: 64 combinacoes",
            len(aj.ParameterGrid(aj.GRADES["mlp"])) == 64)


def troca_na_busca():
    original = aj.REGISTRO
    aj.REGISTRO = Path(tempfile.mkdtemp()) / "busca_em_grade.csv"
    try:
        linha = aj.registra("xgboost", False, {}, 1, np.array([0, 1, 2, 10]),
                            np.array([-2.0, 1.5, 2.0, 7.0]), 0)
    finally:
        aj.REGISTRO = original
    confere("busca: MAE medido com o negativo trocado por zero", linha.mae == 0.875)
    confere("busca: MAE sem a troca e negativos tambem gravados",
            linha.mae_sem_troca == 1.375 and linha.negativos == 1)


def base_real():
    bases = {c: rm.le(c) for c in ("treino", "validacao", "teste")}
    confere("102.383 internacoes nos tres conjuntos", sum(map(len, bases.values())) == 102383)
    treino = bases["treino"]
    procedimentos, formas, municipios = grupos = rm.aprende_grupos(treino)
    confere("403 procedimentos, iguais pelo outro caminho", len(procedimentos) == 403
            and procedimentos == frequentes_por_outro_caminho(treino.PROC_SOLIC))
    raros = {p[:6] for p in treino.PROC_SOLIC if p not in procedimentos}
    confere("87 formas, iguais pelo outro caminho", len(formas) == 87 and formas == raros)
    confere("47 municipios, iguais pelo outro caminho", len(municipios) == 47
            and municipios == frequentes_por_outro_caminho(treino.MUNIC_RES))
    x = {c: rm.prepara(d, grupos) for c, d in bases.items()}
    confere("treino sem procedimento desconhecido", (x["treino"].PROC_SOLIC != "desconhecido").all())
    for nome, colunas in (("xgboost", 843), ("regressao_linear", 836), ("mlp", 843)):
        codificador = rm.cria_codificador(nome, x["treino"])
        nomes = list(codificador.get_feature_names_out())
        confere(f"{nome}: {colunas} colunas", len(nomes) == colunas)
        for c, xc in x.items():
            m = codificador.transform(xc)
            confere(f"{nome} {c}: sem valor vazio", m.shape == (len(xc), colunas)
                    and np.isfinite(m).all())
    referencia = rm.cria_codificador("regressao_linear", x["treino"])
    nomes = set(referencia.get_feature_names_out())
    for c in rm.CATEGORICAS:
        comum = Counter(x["treino"][c]).most_common(1)[0][0]
        confere(f"regressao: {c} sem a coluna da mais comum ({comum})",
                f"categoricas__{c}_{comum}" not in nomes)
    idade = rm.cria_codificador("mlp", x["treino"]).transform(x["treino"])[:, -6]
    confere("mlp: idade padronizada no treino", abs(idade.mean()) < 1e-9
            and abs(idade.std() - 1) < 1e-9)


def ponta_a_ponta():
    registro_real = rm.REGISTRO
    antes = registro_real.stat().st_mtime if registro_real.exists() else None
    treino, validacao = tabela_inventada(2000, 1), tabela_inventada(500, 2)
    validacao.loc[3, "CNES"] = "9999999"
    validacao.loc[4, "PROC_SOLIC"] = "0505050505"
    pasta = Path(tempfile.mkdtemp())
    treino.to_parquet(pasta / "treino.parquet")
    validacao.to_parquet(pasta / "validacao.parquet")
    rm.SPLITS, rm.REGISTRO = pasta, pasta / "experimentos.csv"
    rm.main(list(rm.MODELOS))
    rm.main(list(rm.MODELOS))
    gravado = pd.read_csv(rm.REGISTRO, sep=";", decimal=",", skiprows=[1])
    confere("duas rodadas gravam 8 linhas", len(gravado) == 8)
    metricas = ["mae", "rmse", "r2"]
    confere("mesma semente, mesmo resultado",
            np.array_equal(gravado[metricas][:4].to_numpy(), gravado[metricas][4:].to_numpy()))
    grupos = rm.aprende_grupos(treino)
    xt, xv = rm.prepara(treino, grupos), rm.prepara(validacao, grupos)
    real = validacao.los.to_numpy()
    for nome, modelo in rm.MODELOS.items():
        codificador = rm.cria_codificador(nome, xt)
        erro = modelo.fit(codificador.transform(xt), treino.los).predict(codificador.transform(xv)) - real
        conta = [np.abs(erro).mean(), np.sqrt((erro ** 2).mean()),
                 1 - (erro ** 2).sum() / ((real - real.mean()) ** 2).sum()]
        linha = gravado[gravado.modelo == nome].iloc[0]
        confere(f"{nome}: MAE, RMSE e R2 iguais a conta feita a mao",
                all(abs(round(v, 4) - linha[k]) < 1e-9 for v, k in zip(conta, metricas)))
    confere("regressao linear acerta a regra inventada",
            gravado[gravado.modelo == "regressao_linear"].mae.iloc[0] == 0)
    rm.main(["regressao_linear"], log=True)
    com_log = pd.read_csv(rm.REGISTRO, sep=";", decimal=",", skiprows=[1]).iloc[-1]
    todas = pd.read_csv(rm.REGISTRO, sep=";", dtype=str, keep_default_na=False)
    confere("faixa da literatura so na primeira linha",
            todas.experimento.iloc[0] == "literatura" and (todas.experimento == "literatura").sum() == 1
            and todas.mae.iloc[0] == rm.LITERATURA["mae"])
    codificador = rm.cria_codificador("regressao_linear", xt)
    modelo = rm.MODELOS["regressao_linear"].fit(codificador.transform(xt), np.log1p(treino.los))
    erro = np.expm1(modelo.predict(codificador.transform(xv))) - real
    confere("alvo em log: MAE em dias igual a conta feita a mao",
            abs(round(np.abs(erro).mean(), 4) - com_log.mae) < 1e-9 and com_log.observacao == "alvo em log")
    padrao = rm.cria_codificador("xgboost", xt)
    cnes = [i for i, n in enumerate(padrao.get_feature_names_out()) if n.startswith("categoricas__CNES")]
    confere("hospital que nao existe no treino fica com o bloco zerado",
            not padrao.transform(xv)[3, cnes].any())
    confere("procedimento sem forma no treino vira desconhecido", xv.PROC_SOLIC[4] == "desconhecido")
    depois = registro_real.stat().st_mtime if registro_real.exists() else None
    confere("registro de experimentos do projeto nao foi tocado", antes == depois)


if __name__ == "__main__":
    regras_inventadas()
    relu_na_saida()
    troca_na_busca()
    base_real()
    ponta_a_ponta()
    print(f"{placar['ok']} ok, {placar['falhas']} falhas")
    sys.exit(1 if placar["falhas"] else 0)
