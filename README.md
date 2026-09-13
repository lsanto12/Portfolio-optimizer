# Befektetési portfólió optimalizálása Monte Carlo-szimulációval és PID-vezérelt neurális hálóval

Kurzusbeadandó — Szegedi Tudományegyetem, Bolyai Intézet, 2025.

19 befektetési alapból állít össze portfóliósúlyokat: a jövőbeli árfolyamokat
korrelált Monte Carlo-szimulációval becsli, az optimális súlyvektort pedig egy
neurális háló keresi meg, amelynek gradienslépéseit egy PID-szabályozó hangolja.

---

## A probléma

19 eszköz között 1000 diszkrét egységnyi tőkét elosztani — a „csillagok és rácsok"
összeszámlálás szerint

```
C(n+m-1, n-1) = C(1018, 18) ≈ 1.85 · 10^38
```

lehetséges felosztás. Ekkora teret kimerítő rácsos kereséssel nem lehet bejárni, ezért
a súlyokat egy folytonos, differenciálható paraméterezésben keresem, gradiens alapú
optimalizálással.

## Felépítés

### 1. Szimulációs mag

- Napi loghozamok az alapok nettó eszközérték-idősoraiból
- Kovariancia-mátrix, majd Cholesky-felbontás: `Σ = L·Lᵀ`
- Geometriai Brown-mozgás korrelált piaci sokkokkal:

  ```
  ln(S_t+1) = ln(S_t) + (μ − σ²/2)·Δt + L·Z·√Δt
  ```

  Az `L` mátrix alakítja a független standard normális `Z` vektort korrelált sokkokká.
- Négy hozam-forgatókönyv (stressz / kedvezőtlen / mérsékelt / kedvező)

### 2. `src/cuda_crn_twostage_grid.py` — rácsos keresés (referencia)

A brute force megközelítés GPU-ra optimalizált változata, elsősorban összehasonlítási
alapnak. Amivel a naivnál használhatóbb:

- **CRN (common random numbers)**: minden súlyjelöltet ugyanazon a véletlensorozaton
  értékel ki, így a jelöltek közti különbség nem tűnik el a Monte Carlo-zajban
- **Antitetikus változók** a becslés szórásának további csökkentésére
- **Kétlépcsős szűrés**: sok jelölt kevés szimulációval, majd a legjobbak újraértékelése
  nagy szimulációszámmal
- Metrikák: várható hozam, medián, szórás, Sharpe-hányados, CVaR

Még ezekkel együtt is csak a keresési tér elenyésző töredékét járja be — ez a motiváció
a második megközelítésre.

### 3. `src/nn_portfolio_pid.py` — neurális optimalizáló PID-szabályozással

- A súlyvektort egy neurális háló állítja elő, softmax kimenettel (így a súlyok
  automatikusan nemnegatívak és 1-re összegződnek)
- **Moduláris loss**: várható hozam / medián / Sharpe / CVaR / entrópia (diverzifikáció) /
  maxsúly-büntetés — mindegyik külön kapcsolható és súlyozható parancssorból
- **PID-szabályozás**: két hurok fut, egy a választott metrikára és egy magára a loss-ra.
  A `u(t) = Kp·e(t) + Ki·∫e dt + Kd·de/dt` kimenetükből lesz egy skálázó tényező, ami a
  loss-ra kerül a visszaterjesztés előtt. A cél az volt, hogy a lépéshossz automatikusan
  igazodjon: nagy hiba esetén erősebb, a cél közelében óvatosabb korrekció.

  Az ötlet egy repülőgép-vezérlésről szóló dokumentumfilmből jött — a szabályozástechnikai
  PID-hurok és a gradiens alapú tanulás analógiája.

## Futtatás

```bash
pip install -r requirements.txt

# 1) Kovariancia-mátrix előállítása az idősorokból
python src/build_cov.py --data-dir data --out data/hold_alapok_osszefuzve.xlsx

# 2) Neurális + PID optimalizáló
python src/nn_portfolio_pid.py \
    --scenario mersekelt --epochs 50000 \
    --use-sharpe 1 --use-cvar 1 --use-pid-loss 1

# 3) Rácsos keresés (összehasonlításhoz)
python src/cuda_crn_twostage_grid.py \
    --scenario mersekelt --two-stage --crn --antithetic \
    --candidates 3003 --sims-per-eval 10000 \
    --out results/weight_search_cuda.csv
```

Minden paraméter `--help`-pel listázható. CUDA-képes GPU ajánlott; CPU-n is fut, lassabban.

**Futásidő:** RTX 4050 és RTX 3080 kártyán is néhány perc a fenti beállításokkal.

## Könyvtárszerkezet

```
src/
  build_cov.py                 kovariancia-mátrix az idősorokból
  cuda_crn_twostage_grid.py    GPU-s rácsos keresés (CRN, kétlépcsős)
  nn_portfolio_pid.py          neurális optimalizáló PID-szabályozással
data/                          alapok napi nettó eszközérték-idősorai
results/                       súlyvektorok és metrikák
docs/prezentacio.pdf           a projektet bemutató diasor
```

## Adatok

A `data/` mappában két alap idősora szerepel mintaként (2017-07-17 – 2025-11-14,
2082 közös kereskedési nap). Az alapadatok nyilvánosak, a formátumra hozás és az
összefűzés saját munka. A várható hozamok (μ) a négy forgatókönyvhöz a kibocsátó
publikált forgatókönyv-adataiból származnak.

## Technológiák

Python · PyTorch (CUDA) · NumPy · pandas · openpyxl
