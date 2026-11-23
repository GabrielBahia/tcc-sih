import pandas as pd

import config

CONJUNTOS = {"treino": config.SPLIT_TREINO, "validacao": config.SPLIT_VALIDACAO,
             "teste": config.SPLIT_TESTE}


def main():
    df = pd.read_parquet(config.TRATADO / "sih_jf_tratado.parquet")
    entrada = pd.to_datetime(df["DT_INTER"], format="%Y%m%d")
    config.SPLITS.mkdir(parents=True, exist_ok=True)
    for nome, (inicio, fim) in CONJUNTOS.items():
        parte = df[entrada.between(inicio, fim)]
        parte.to_parquet(config.SPLITS / f"{nome}.parquet", index=False)
        print(f"{nome}: {len(parte)} internacoes, de {inicio} a {fim}")


if __name__ == "__main__":
    main()
