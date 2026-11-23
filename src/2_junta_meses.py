import pandas as pd

import config


def main():
    partes = []
    for arquivo in sorted(config.COMPETENCIAS.glob("*.parquet")):
        df = pd.read_parquet(arquivo)
        if len(df):
            df.insert(0, "COMPETENCIA", f"20{arquivo.stem[4:6]}-{arquivo.stem[6:8]}")
            partes.append(df)
    completo = pd.concat(partes, ignore_index=True, sort=False)

    entrada = pd.to_datetime(completo["DT_INTER"], format="%Y%m%d", errors="coerce")
    inicio, fim = config.JANELA_INTERNACAO
    janela = completo[entrada.between(inicio, fim)]

    config.CONSOLIDADO.mkdir(parents=True, exist_ok=True)
    completo.to_parquet(config.CONSOLIDADO / "sih_jf_completo.parquet", index=False)
    janela.to_parquet(config.CONSOLIDADO / "sih_jf_janela.parquet", index=False)
    print(f"{len(partes)} meses, {len(completo)} registros, {len(janela)} na janela de {inicio} a {fim}")


if __name__ == "__main__":
    main()
