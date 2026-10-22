import pandas as pd

# Cargar el archivo recién descargado por DVC
path = r"data\processed\master_weekly_screening.parquet"
try:
    df = pd.read_parquet(path)
except FileNotFoundError:
    print("Error: No se encontró el dataset maestro. Verifica que DVC haya descargado el archivo.")
    exit()

print(f"Dimensiones del dataset: {df.shape}")
print("Columnas disponibles:", df.columns.tolist())

# Filtrar el nuevo municipio piloto: Villavicencio (Código 50001)
vvc = df[df["municipality_code"].astype(str).str.zfill(5) == "50001"].copy()

print(f"\nRegistros de Villavicencio: {len(vvc)} semanas")
print("Carga total de casos en el periodo (confirmados):", vvc["confirmed_records_screening"].sum())
print("\nPrimeras filas de Villavicencio:")
print(vvc.head())