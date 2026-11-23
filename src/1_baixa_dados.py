import time
from ftplib import FTP

import datasus_dbc
import pandas as pd
from dbfread import DBF

import config


def competencias():
    ano, mes = config.COMPETENCIA_INICIAL
    lista = []
    while (ano, mes) <= config.COMPETENCIA_FINAL:
        lista.append((ano, mes))
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)
    return lista


def baixa(nome, destino, tentativas=3):
    for tentativa in range(1, tentativas + 1):
        try:
            with FTP(config.FTP_HOST, timeout=300) as ftp:
                ftp.login()
                ftp.set_pasv(True)
                ftp.cwd(config.FTP_DIR)
                parcial = destino.with_suffix(".parcial")
                with open(parcial, "wb") as f:
                    ftp.retrbinary("RETR " + nome, f.write)
            parcial.replace(destino)
            return True
        except Exception as erro:
            print(f"   tentativa {tentativa} falhou ({erro})")
            time.sleep(5 * tentativa)
    return False


def so_juiz_de_fora(caminho_dbc):
    caminho_dbf = caminho_dbc.with_suffix(".dbf")
    datasus_dbc.decompress(str(caminho_dbc), str(caminho_dbf))
    tabela = DBF(str(caminho_dbf), encoding="latin-1", load=False)
    colunas = [campo.name for campo in tabela.fields]
    linhas, total = [], 0
    for registro in tabela:
        total += 1
        if str(registro.get("MUNIC_MOV", "")).strip() == config.MUNICIPIO_JF:
            linhas.append({c: "" if registro.get(c) is None else str(registro.get(c)).strip()
                           for c in colunas})
    caminho_dbf.unlink()
    return pd.DataFrame(linhas, columns=colunas), total


def processa(ano, mes):
    nome = f"RD{config.UF}{ano % 100:02d}{mes:02d}.dbc"
    linha = {"competencia": f"{ano}-{mes:02d}", "arquivo": nome, "situacao": "ja existia",
             "linhas_jf": None, "linhas_mg": None}
    parquet = config.COMPETENCIAS / f"{nome[:-4]}.parquet"
    if parquet.exists():
        return linha
    dbc = config.DBC / nome
    if not (dbc.exists() and dbc.stat().st_size > 0) and not baixa(nome, dbc):
        return {**linha, "situacao": "falhou o download"}
    df, total = so_juiz_de_fora(dbc)
    df.to_parquet(parquet, index=False)
    return {**linha, "situacao": "ok", "linhas_jf": len(df), "linhas_mg": total}


def main():
    for pasta in (config.DBC, config.COMPETENCIAS, config.LOGS):
        pasta.mkdir(parents=True, exist_ok=True)
    log = []
    for ano, mes in competencias():
        linha = processa(ano, mes)
        log.append(linha)
        print(f"{linha['arquivo']}: {linha['situacao']}")
    log = pd.DataFrame(log)
    log.to_csv(config.LOGS / "extracao_datasus.csv", index=False, encoding="utf-8")
    print(log.situacao.value_counts().to_string())


if __name__ == "__main__":
    main()
