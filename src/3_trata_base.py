import json
import re

import pandas as pd

import config

JANELA_INICIO, JANELA_FIM = (pd.Timestamp(d) for d in config.JANELA_INTERNACAO)
CONTINUACAO = {str(c) for c in range(21, 29)} | {"51"}
HOSPITAIS_EXCLUIDOS = {"2153742", "3005410"}
PREDITORAS = ["CAR_INT", "CNES", "DT_INTER", "ESPEC", "MUNIC_RES", "NAT_JUR", "NASC",
              "PROC_SOLIC", "RACA_COR", "SEXO", "DIAG_PRINC"]
COLUNAS = ["N_AIH"] + PREDITORAS + ["idade", "cid_item", "CEP", "continuacoes", "los"]
CID = re.compile(r"^([A-Z])(\d{2})(?:\.(\d))?$")


def texto(serie):
    return serie.fillna("").astype(str).str.strip()


def data(serie):
    return pd.to_datetime(texto(serie), format="%Y%m%d", errors="coerce")


def registra(passos, regra, df, antes):
    passos.append({"regra": regra, "antes": antes, "saiu": antes - len(df), "depois": len(df),
                   "los_medio": round(df["los"].mean(), 2), "los_mediano": df["los"].median(),
                   "los_max": int(df["los"].max())})
    return len(df)


def junta_continuacoes(completo, principais):
    cont = completo[texto(completo["IDENT"]) == "5"].copy()
    cont = cont[cont["N_AIH"].isin(set(principais["N_AIH"]))]
    cont["dt_saida"] = data(cont["DT_SAIDA"])
    cont = cont.sort_values("dt_saida")
    return cont.groupby("N_AIH").agg(dt_saida_final=("dt_saida", "max"),
                                     morte_final=("MORTE", "last"),
                                     continuacoes=("N_AIH", "size"))


def aih_da_base_completa(completo):
    aih = completo[["N_AIH", "DT_INTER", "DT_SAIDA", "COBRANCA", "NASC", "SEXO", "CEP", "CNES"]].copy()
    aih["ini"] = data(aih["DT_INTER"])
    aih["fim"] = data(aih["DT_SAIDA"])
    aih = aih[aih["fim"] >= JANELA_INICIO - pd.Timedelta(days=1095)]
    aih["cob"] = texto(aih["COBRANCA"])
    aih["chave"] = texto(aih["NASC"]) + "|" + texto(aih["SEXO"]) + "|" + texto(aih["CEP"])
    return (aih.sort_values(["N_AIH", "fim"]).groupby("N_AIH")
            .agg(ini=("ini", "min"), fim=("fim", "max"), cob=("cob", "last"),
                 chave=("chave", "first"), CNES=("CNES", "first"))
            .reset_index())


def pares(aih):
    esq = aih.loc[aih.cob.isin(CONTINUACAO), ["N_AIH", "chave", "CNES", "fim"]]
    par = esq.merge(aih[["N_AIH", "chave", "CNES", "ini"]], on=["chave", "CNES"], suffixes=("_a", "_b"))
    par = par[par.N_AIH_a != par.N_AIH_b]
    return par[(par.ini - par.fim).dt.days.between(0, 1)]


def cadeias(aih):
    posicao = {n: i for i, n in enumerate(aih.N_AIH)}
    pai = list(range(len(aih)))

    def raiz(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    par = pares(aih)
    for a, b in zip(par.N_AIH_a, par.N_AIH_b):
        ra, rb = raiz(posicao[a]), raiz(posicao[b])
        if ra != rb:
            pai[rb] = ra
    aih = aih.copy()
    aih["cadeia"] = [raiz(i) for i in range(len(aih))]
    return aih


def cruzam_a_borda(aih, numeros):
    aih = cadeias(aih)
    aih["na_janela"] = aih.N_AIH.isin(set(numeros))
    cadeia = aih.groupby("cadeia").agg(total=("N_AIH", "size"), dentro=("na_janela", "sum"))
    partida = cadeia[(cadeia.dentro > 0) & (cadeia.total > cadeia.dentro)]
    fora = aih[aih.cadeia.isin(partida.index) & aih.na_janela]
    aberta = aih[aih.na_janela & (aih.fim > JANELA_FIM) & aih.cob.isin(CONTINUACAO)]
    return set(fora.N_AIH) | set(aberta.N_AIH)


def sem_desfecho(aih, numeros):
    tem_seguinte = set(pares(aih).N_AIH_a)
    sem = aih[aih.cob.isin(CONTINUACAO) & ~aih.N_AIH.isin(tem_seguinte)]
    return set(sem.N_AIH) & set(numeros)


def chave_da_lista(codigo, sub_padrao):
    letra, numero, sub = CID.match(codigo.strip()).groups()
    return letra, int(numero), int(sub) if sub is not None else sub_padrao


def chave_do_cid(codigo):
    codigo = str(codigo).strip().upper()
    letra, numero, sub = codigo[0:1], codigo[1:3], codigo[3:4]
    if len(codigo) < 3 or not letra.isalpha() or not numero.isdigit():
        return None
    return letra, int(numero), int(sub) if sub.isdigit() else -1


def lista_de_morbidade():
    itens = json.loads((config.AUXILIAR / "lista_morbidade.json").read_text(encoding="utf-8"))
    faixas = []
    for item, _, codigos in itens:
        if "." in item:
            continue
        for parte in filter(None, (p.strip() for p in codigos.split(","))):
            inicio, fim = parte.split("-") if "-" in parte else (parte, parte)
            faixas.append((chave_da_lista(inicio, -1), chave_da_lista(fim, 9), item))
    return sorted(faixas)


def item_do_cid(codigos):
    faixas = lista_de_morbidade()
    mapa = {}
    for codigo in codigos:
        k = chave_do_cid(codigo)
        if k is not None:
            mapa[codigo] = next((item for inicio, fim, item in faixas if inicio <= k <= fim), None)
    return {c: i for c, i in mapa.items() if i is not None}


def main():
    completo = pd.read_parquet(config.CONSOLIDADO / "sih_jf_completo.parquet")
    df = pd.read_parquet(config.CONSOLIDADO / "sih_jf_janela.parquet")
    df["dt_inter"] = data(df["DT_INTER"])
    df["dt_saida"] = data(df["DT_SAIDA"])
    df["los"] = (df["dt_saida"] - df["dt_inter"]).dt.days
    df["morte"] = texto(df["MORTE"])
    passos = []
    n = registra(passos, "0. base da janela", df, len(df))

    df = df[texto(df["IDENT"]) == "1"].copy()
    ultimo = junta_continuacoes(completo, df)
    df = df.join(ultimo, on="N_AIH")
    alvo = df["dt_saida_final"].notna()
    df.loc[alvo, "los"] = (df.loc[alvo, "dt_saida_final"] - df.loc[alvo, "dt_inter"]).dt.days
    df.loc[alvo, "morte"] = texto(df.loc[alvo, "morte_final"])
    df["continuacoes"] = df["continuacoes"].fillna(0).astype(int)
    n = registra(passos, "1. consolidar AIH de continuacao", df, n)

    aih = aih_da_base_completa(completo)
    df = df[~df["N_AIH"].isin(cruzam_a_borda(aih, df["N_AIH"]))]
    n = registra(passos, "2. retirar internacoes que cruzam a borda", df, n)

    df = df[~df["N_AIH"].isin(sem_desfecho(aih, df["N_AIH"]))]
    n = registra(passos, "3. excluir internacoes sem desfecho", df, n)

    df = df[~texto(df["CNES"]).isin(HOSPITAIS_EXCLUIDOS)]
    n = registra(passos, "4. excluir hospitais de permanencia constante", df, n)

    df = df[df["morte"] != "1"]
    n = registra(passos, "5. excluir obitos", df, n)

    df["idade"] = (df["dt_inter"] - data(df["NASC"])).dt.days / 365.25
    df = df[df["idade"].between(0, 120)]
    n = registra(passos, "6. recalcular idade pelo NASC", df, n)

    df["cid_item"] = texto(df["DIAG_PRINC"]).map(item_do_cid(texto(df["DIAG_PRINC"]).unique()))
    n = registra(passos, "7. agrupar o CID na Lista de Morbidade", df, n)

    df = df[COLUNAS]
    registra(passos, "8. manter so as colunas da admissao", df, n)

    config.TRATADO.mkdir(parents=True, exist_ok=True)
    (config.RESULTADOS / "tabelas").mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.TRATADO / "sih_jf_tratado.parquet", index=False)
    passos = pd.DataFrame(passos)
    passos.to_csv(config.RESULTADOS / "tabelas" / "tratamento_passo_a_passo.csv", index=False,
                  encoding="utf-8-sig", sep=";", decimal=",")
    print(passos.to_string(index=False))


if __name__ == "__main__":
    main()
