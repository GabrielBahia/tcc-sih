from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DADOS = RAIZ / "dados"
DBC = DADOS / "00_dbc"
COMPETENCIAS = DADOS / "01_competencias"
CONSOLIDADO = DADOS / "02_consolidado"
TRATADO = DADOS / "03_tratado"
SPLITS = DADOS / "04_splits"
AUXILIAR = DADOS / "05_auxiliar"
LOGS = RAIZ / "logs"
RESULTADOS = RAIZ / "resultados"

FTP_HOST = "ftp.datasus.gov.br"
FTP_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados/"

UF = "MG"
MUNICIPIO_JF = "313670"

COMPETENCIA_INICIAL = (2008, 1)
COMPETENCIA_FINAL = (2026, 6)

JANELA_INTERNACAO = ("2024-01-01", "2025-12-31")

SPLIT_TREINO = ("2024-01-01", "2024-12-31")
SPLIT_VALIDACAO = ("2025-01-01", "2025-06-30")
SPLIT_TESTE = ("2025-07-01", "2025-12-31")
