"""Umbrales de las reglas de conteo de D9 (02 §8).
 
Si dos opciones fueran equivalentes, ganar cada casilla (o cada inicio discrepante)
sería una moneda con p = 0,5. El umbral es el menor número de victorias cuya
probabilidad por azar, en la dirección declarada (una cola), es menor que 5 %.
 
Resultado esperado:
  prueba (18 casillas): 13 (4,8 %); desarrollo (24 casillas): 17 (3,2 %);
  McNemar exacto de una cola, umbral por número de discrepantes d:
  d < 5 imposible; 5→5, 6→6, 7→7, 8→7, 9→8, 10→9;
  4 de 4 años: 1/16 = 6,25 %.
También imprime la potencia si la opción evaluada gana de verdad el 65, 75 u 85 %
de las casillas.
 
Uso: python -I umbrales_d9.py
"""
from __future__ import annotations
 
from scipy.stats import binom
 
ALPHA = 0.05
 
 
def threshold(n: int, alpha: float = ALPHA) -> int | None:
    """Menor k con P(X >= k | n, 0,5) < alpha."""
    return next((k for k in range(n + 1) if binom.sf(k - 1, n, 0.5) < alpha), None)
 
 
def main() -> int:
    for label, n in (("prueba", 18), ("desarrollo", 24)):
        k = threshold(n)
        chance = binom.sf(k - 1, n, 0.5)
        power = ", ".join(f"p={p}: {binom.sf(k - 1, n, p):.2f}" for p in (0.65, 0.75, 0.85))
        print(f"{label}: {k} de {n} (azar {chance:.3f}); potencia {power}")
        if n == 18:
            for lo in (10, 11, 12):
                print(f"  {lo} de 18 o más por azar: {binom.sf(lo - 1, n, 0.5):.3f}")
    print("McNemar una cola:", {d: threshold(d) for d in range(1, 11)})
    print("4 de 4 años por azar:", 0.5 ** 4)
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())