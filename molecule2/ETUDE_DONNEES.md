# Étude des données — Enveda CASMI 2026 (identification de molécules à partir de spectres MS/MS)

> Analyse réalisée le 05/10/2026 sur `train.parquet` (2 539 608 lignes), `test.parquet` (1 213 lignes) et `sample_submission.csv`.
> Les statistiques sur les listes de pics viennent d'un échantillon stratifié de 3 % du train (76 184 spectres). Les statistiques chimiques viennent de RDKit (15 000 molécules par groupe).

---

## 1. Le problème en une phrase

On reçoit **un ou plusieurs spectres de masse en tandem (MS/MS)** d'une molécule inconnue et on doit proposer **jusqu'à 25 structures candidates (SMILES)**, de la plus probable à la moins probable.

- **Métrique : MRR@25** (Mean Reciprocal Rank). Si la bonne structure est en position k, la molécule rapporte 1/k (1.0 en 1re position, 0.5 en 2e, 0.04 en 25e, 0 si elle est absente).
- **Une prédiction est correcte** si son **InChIKey14** (les 14 premiers caractères de l'InChIKey, après canonicalisation des tautomères par RDKit) est identique à celui de la vraie molécule. Ce bloc ne code que la **connectivité 2D** : **la stéréochimie et la tautomérie ne comptent pas**.
  → Cela n'apporte rien de proposer deux stéréoisomères ou deux tautomères d'une même molécule : ils prendraient deux places pour le même InChIKey14. Il faut **dédoublonner les candidats par InChIKey14**.
- **Format de soumission** : `molecule_id,smiles` avec 25 SMILES séparés par `;` (400 lignes, une par molécule de test).

### Rappel physique (pour bien lire les features)

1. La molécule neutre **M** est ionisée (ESI) en un **ion précurseur** grâce à un **adduit** : `[M+H]+` (on ajoute un proton), `[M-H]-` (on en retire un), `[M+Na]+`, etc.
2. Le spectromètre mesure le rapport masse/charge de cet ion : **`precursor_mz`**.
3. L'ion est isolé puis fragmenté par collision avec un gaz. L'**énergie de collision** (CE) règle l'intensité de la fragmentation.
4. On mesure les fragments, ce qui donne une liste de couples **(m/z, intensité)** : c'est le **spectre MS2**, l'« empreinte » de la structure.

---

## 2. Architecture des fichiers

```
enveda-CASMI26-molecule-id-mass-spectra/
├── train.parquet          2,8 Go  | 2 539 608 spectres | 275 810 molécules (InChIKey14) | 19 colonnes
├── test.parquet           4,8 Mo  | 1 213 spectres     | 400 molécules                  | 12 colonnes
└── sample_submission.csv  400 lignes : molecule_id, smiles (25 SMILES séparés par ';')
```

**Granularité** : **une ligne correspond à un spectre**, pas à une molécule. Une molécule a plusieurs spectres (énergies, adduits ou polarités différents).

| | Train | Test |
|---|---|---|
| Clé molécule | `inchikey14` (aussi `inchikey`, `normalized_smiles`) | `molecule_id` (anonymisé, ex. `m_005e53`) |
| Clé spectre | aucune (index de ligne) | `spectrum_id` |
| Spectres / molécule | médiane 6, Q90 = 14, max 1 468 | 1 à 9, médiane 3 |
| Cible | `normalized_smiles` / `inchikey14` | à prédire |

**Organisation physique du train** : le fichier contient 21 *row groups* triés par source (`enveda-180` occupe les groupes 0 à 8, puis gnps, massbank, pluskal, riken, spectraverse, et `enveda-np-examples` dans le dernier). Avec `pyarrow.ParquetFile.read_row_group(i)`, on peut traiter le fichier par morceaux sans charger les 2,8 Go en RAM. Chargez les colonnes scalaires seules : elles tiennent facilement en mémoire, ce sont les listes de pics qui pèsent lourd.

---

## 3. Dictionnaire des features

### 3.1 Colonnes communes au train et au test (= les entrées du modèle)

| Feature | Type | Interprétation | Valeurs observées / remarques |
|---|---|---|---|
| `ms2_mzs` | `list<float64>` | Rapports masse/charge des pics de fragments (en Th ≈ Da pour z=1). Triés par ordre croissant, sans doublons. | Train : médiane 42 pics, max 73 318. **Test : médiane 230 pics** (Q25 = 101, Q75 = 443, max 3 259). |
| `ms2_normalized_intensities` | `list<float64>` | Intensité de chaque pic, **normalisée par le pic de base (max = 1)**, alignée avec `ms2_mzs`. | Le max vaut toujours 1. Il y a beaucoup de pics minuscules : la médiane des intensités est 4e-4 (train) et **1,7e-4 (test)**. Les spectres de test sont plus bruités et plus denses. |
| `base_peak_intensity` | `float64` | Intensité absolue du pic le plus fort, en unités instrument. Elle reflète la quantité de matière ou la qualité du signal. | **Manquante pour 46 % du train** (toutes les sources publiques sauf gnps). Complète en test. Elle dépend de l'instrument : une feature à manier avec prudence. |
| `precursor_mz` | `float64` | m/z de l'ion précurseur. Avec l'adduit, elle donne la **masse exacte de la molécule** (feature la plus discriminante). | Train : médiane 350, mais valeurs aberrantes (min 2, max 2,4e6). **Test : 245 à 460**, médiane 329. |
| `adduct` | `string` (catégorielle) | Forme ionisée, normalisée par les organisateurs. | 121 modalités en train. **Test : 7 seulement** : `[M+H]+` 959, `[M-H]-` 193, `[M+CH2O2-H]-` (formiate) 31, `[M+Na]+` 22, `[M+NH4]+` 4, `[M+K]+` 2, `[M+Cl]-` 2. |
| `ionization_mode` | `string` (binaire) | Polarité : `positive` / `negative`. Redondante avec le signe de l'adduit. | Train : 75 % positif. Test : 81 % positif. **97 molécules de test ont des spectres dans les deux polarités.** |
| `instrument_type` | `string` (catégorielle, sale) | Type d'analyseur de masse. | Train : 83 libellés non harmonisés (`QTOF`, `qTof`, `qtof`, `LC-ESI-QTOF`…), 33 053 valeurs nulles, plus des `"nan"` et `""`. **Test : 100 % `timsTOF`.** |
| `collision_energy_orig` | `string` | Énergie de collision **telle que déclarée par la source**, au format brut. | Exemples : `"20,40,60"`, `"-40"`, `"[20.0, 30.0]"`, `"35HCD"`, `"Ramp 5-60 V"`. Le signe `-` indique simplement le mode négatif. Valeur nulle pour 320 k lignes. |
| `collision_energy_ev` | `list<float64>` | Version **harmonisée en eV** (toujours positive). Une liste de plusieurs valeurs indique un spectre « fusionné » ou acquis en *stepped CE*. | Test : `[20]`, `[40]`, `[60]` ou `[20,40,60]`. Les valeurs NCE (Orbitrap) ont été converties en eV, médiane 19. Liste vide quand l'unité est `unknown`. |
| `collision_energy_orig_units` | `string` | Unité d'origine : `eV`, `NCE` (énergie normalisée Orbitrap), `V`, `unknown`. | Test : 100 % `eV`. |

### 3.2 Colonnes du train uniquement (cibles et métadonnées)

| Feature | Type | Interprétation | Remarques |
|---|---|---|---|
| `normalized_smiles` | `string` | **Cible** : structure de la molécule, au format SMILES normalisé. | 277 566 SMILES pour 275 810 InChIKey14 : quelques molécules ont 2 à 5 SMILES (stéréo ou tautomères). |
| `inchikey` | `string` | Identifiant InChIKey complet sur 27 caractères : connectivité, stéréo et protonation. | 139 InChIKey14 ont plusieurs InChIKey complets (stéréoisomères). |
| `inchikey14` | `string` | **Clé d'évaluation**. Utilisez-la pour regrouper les spectres par molécule et pour construire les splits de validation. | 275 810 molécules uniques. |
| `molecular_formula` | `string` | Formule brute (ex. `C20H15N3O2`). | 51 291 formules distinctes, donc **en moyenne environ 5 isomères par formule**. Éléments : C, H, O, N, S, F, Cl, Br, P, I, et des traces de B, Si, As, Se, Sn. |
| `ingest_lib` | `string` | Bibliothèque d'origine du spectre. | 11 sources, voir la section 4. |
| `adduct_orig` | `string` | Adduit tel qu'écrit dans la source. | Synonyme brut de `adduct` (ex. `[M+H4N]+` devient `[M+NH4]+`). **Inutile** une fois `adduct` disponible. |
| `precursor_error_ppm` | `float64` | Écart, en ppm, entre le `precursor_mz` mesuré et le m/z théorique (formule + adduit). Indicateur de qualité. | Médiane \|err\| = 1,5 ppm, Q90 = 5 ppm, **mais 6,9 % des spectres dépassent 10 ppm** et des valeurs absurdes montent jusqu'à 4,7e8. **Filtre de nettoyage recommandé : \|err\| < 10–20 ppm.** |
| `num_peaks` | `int64` | Nombre de pics (= `len(ms2_mzs)`). | Absente du test, mais elle se recalcule directement. |

---

## 4. Les sources (`ingest_lib`) : le point le plus important pour comprendre le dataset

Le train est un **assemblage hétérogène** de bibliothèques, et une seule d'entre elles ressemble vraiment au test.

| Source | Spectres | Molécules | % positif | Instrument | Pics (médiane) | \|ppm\| médian | Unité CE | Nature |
|---|---:|---:|---:|---|---:|---:|---|---|
| **enveda-180** | 1 153 785 (45 %) | 182 941 | 77 % | **timsTOF** | 138 | 0,98 | eV | Molécules **synthétiques** (chimiothèque) |
| pluskal_ms2 | 527 581 | 46 821 | 61 % | Orbitrap | 21 | 1,66 | NCE | MSnLib (Pluskal lab) |
| riken | 347 171 | 15 892 | 90 % | Orbitrap / QTOF | 9 | 3,75 | NCE | RIKEN PlaSMA / MS-DIAL |
| gnps | 220 849 | 45 750 | 81 % | variés | 46 | 1,67 | unknown | Bibliothèque communautaire GNPS |
| massbank | 101 727 | 9 180 | 67 % | variés | 11 | 1,95 | eV | MassBank |
| mona | 92 416 | 11 681 | 75 % | variés | 57 | 1,80 | NCE | MoNA |
| spectraverse | 50 933 | 9 631 | 66 % | variés | 47 | 0,65 | unknown | Spectraverse |
| msdial | 40 765 | 9 127 | 58 % | variés | 11 | 1,47 | unknown | Bibliothèques MS-DIAL |
| drug_plus | 2 545 | 2 539 | 100 % | ? | 108 | 1,73 | unknown | Médicaments |
| **enveda-np-examples** | 1 184 | 250 | 67 % | **timsTOF** | 141 | 1,65 | eV | Produits naturels Enveda (même plateforme, mais spectralement moins proche du test qu'enveda-180, voir `01_EDA.ipynb`) |
| masaryk | 652 | 416 | 100 % | ? | 9 | 1,48 | unknown | |

Observations clés :

- **`enveda-180` + `enveda-np-examples` = même plateforme que le test** : timsTOF, CE 20/40/60 eV et fusion `20,40,60`, mêmes adduits majoritaires, densité de pics comparable (130–230 pics). Les sources publiques ont des spectres beaucoup plus « propres » (souvent < 20 pics, déjà filtrés).
- `enveda-180` contient beaucoup de **dimères** (`[2M+Na]+` : 188 k, `[2M+H]+` : 89 k), **absents du test**. À filtrer ou sous-pondérer.
- **Peu de recouvrement entre sources** : seulement 2 288 molécules d'enveda-180 se retrouvent dans une autre source, contre 25 825 molécules partagées au sein des bibliothèques publiques.
- `base_peak_intensity` n'est renseignée que pour enveda-180, enveda-np-examples, gnps et masaryk. Sa présence ou son absence révèle presque la source : **ne pas l'utiliser naïvement comme feature**.

---

## 5. Ce qui distingue le test : domain shift

### 5.1 Côté spectres

| | Train (échantillon) | enveda-180 | **Test** |
|---|---|---|---|
| Pics / spectre (médiane) | 42 | 138 | **230** |
| Intensité médiane d'un pic | 4,2e-4 | – | **1,7e-4** |
| Spectres avec un pic au-delà de précurseur + 1 Da | ~25–45 % | 43 % | **56 %** |
| Instrument | mixte | timsTOF | timsTOF |
| CE | hétérogène | 20/40/60/fusion | 20/40/60/fusion |

Les spectres de test contiennent **beaucoup de bruit de faible intensité** et souvent l'**enveloppe isotopique du précurseur** (pic M+1 à +1,00336 Da, ratio médian M+1/M ≈ 0,17 pour les `[M+H]+`). Ce ratio est une information utile, car il dépend grossièrement du nombre de carbones (environ 1,1 % par C). Environ 3 % des spectres `[M+H]+` ont un M+2 fort, signe probable de la présence de Cl ou Br.

### 5.2 Côté molécules

Masse neutre calculée à partir de `precursor_mz` et de l'adduit : **246 à 439 Da, médiane 328** (fenêtre étroite). La dispersion intra-molécule est inférieure à 0,002 Da, ce qui confirme la cohérence des adduits.

| Descripteur (médianes, RDKit) | enveda-180 (synthétique) | public | enveda-np-examples (NP) |
|---|---:|---:|---:|
| Masse exacte | 330 | 350 | 354 |
| Atomes lourds | 23 | 25 | 25 |
| Nb N | **3** | 2 | **0** |
| Nb O | 2 | 3 | **5** |
| Centres stéréo | 1 | 1 | **2,5** |
| Fsp3 | 0,44 | 0,44 | 0,47 |
| % sans azote | **0 %** | 21 % | **53 %** |
| % halogénés | **36 %** | 20 % | 5 % |
| % masse nominale impaire (nombre impair de N) | 50 % | 46 % | 29 % |
| **Test** : % masse nominale impaire | | | **53 %** |

Interprétation : Enveda décrit le test comme des **produits naturels réels ou potentiels et leurs analogues**. Les exemples NP fournis sont oxygénés et pauvres en azote. Pourtant, **53 % des molécules de test ont une masse nominale impaire**, donc un nombre impair d'atomes N (*règle de l'azote*). Le test contient donc une part importante de composés azotés (alcaloïdes, analogues). C'est une feature binaire simple et très informative.

### 5.3 Espace de recherche

Pour **100 % des molécules de test**, il existe au moins une molécule du train de même masse à ±5 ppm (en médiane 55 candidats, 78 à ±10 ppm). Mais **la bonne réponse n'est probablement pas dans le train** (le test n'a jamais été publié). Les isobares du train servent de *voisins structuraux*, pas de réponses. Une base de candidats externe (PubChem, COCONUT pour les produits naturels, etc.) filtrée par masse ou formule sera vraisemblablement nécessaire.

---

## 6. Problèmes de qualité à traiter

1. **`instrument_type`** : 83 libellés à harmoniser en ~6 classes (`timsTOF`, `Orbitrap/QFT/ITFT/HCD`, `QTOF/TOF`, `QQQ`, `IonTrap`, `unknown`).
2. **`precursor_error_ppm`** : valeurs aberrantes (jusqu'à 4,7e8 ppm), 4 286 valeurs nulles. Filtrer \|ppm\| > 20.
3. **`precursor_mz`** : valeurs absurdes (2 Da, 2,4 millions). Filtrer par exemple entre 50 et 2 000, et idéalement entre ~150 et 700 pour coller au test.
4. **Énergies de collision** : 4 unités non comparables (eV, NCE, V, unknown). `collision_energy_ev` est la version harmonisée, mais elle reste vide pour 12,8 % du train. Ajouter un indicateur `ce_missing`.
5. **Adduits rares** : 121 modalités, dont beaucoup à moins de 100 spectres. En test, seuls 7 adduits apparaissent. **Garder ces 7, regrouper ou exclure le reste**, y compris tous les `[2M…]` et multichargés.
6. **Déséquilibre par molécule** : certaines molécules ont jusqu'à 1 468 spectres. Pondérer par 1/n_spectres ou sous-échantillonner pour ne pas biaiser l'entraînement.
7. **Doublons de SMILES** pour un même InChIKey14 : toujours raisonner au niveau InChIKey14.
8. **Pics au-delà du précurseur** (isotopes, bruit) : à garder pour les features isotopiques, mais à retirer pour la similarité spectrale « classique ».

---

## 7. Pistes de feature engineering

### 7.1 Niveau spectre : métadonnées (directement exploitables)

| Feature | Construction | Pourquoi |
|---|---|---|
| `neutral_mass` | `precursor_mz − Δ(adduct)`, avec Δ : `[M+H]+` 1,007276 ; `[M-H]-` −1,007276 ; `[M+Na]+` 22,989218 ; `[M+NH4]+` 18,033823 ; `[M+K]+` 38,963158 ; `[M+Cl]-` 34,969402 ; `[M+CH2O2-H]-` 44,998201 | **Feature n° 1** : contraint la formule brute et filtre les candidats. |
| `mass_defect` | `neutral_mass − floor(neutral_mass)` | Signale la richesse en H (défaut élevé) ou la présence d'halogènes, de S, de P (défaut bas). |
| `odd_nominal_mass` | `round(neutral_mass) % 2` | Règle de l'azote : parité du nombre d'atomes N. |
| `adduct` (one-hot, 7 classes) et `polarity` | | Le schéma de fragmentation dépend de l'adduit (les adduits Na+ fragmentent peu). |
| `ce_min`, `ce_max`, `ce_is_merged` | à partir de `collision_energy_ev` | Une CE basse donne surtout le précurseur, une CE haute de petits fragments. |
| `instrument_group` | harmonisation | Contrôle du domain shift (ou filtre « timsTOF only »). |

### 7.2 Niveau spectre : dérivées des pics

| Feature | Construction | Interprétation |
|---|---|---|
| Nombre de pics après débruitage | pics avec intensité > 0,5–1 % du max | Neutralise la différence de bruit entre sources. |
| Entropie spectrale | `−Σ p log p`, avec `p = I / ΣI` | Mesure si le spectre est « simple » (peu de fragments dominants) ou riche. |
| Fraction d'intensité restante au précurseur | I(pic à precursor_mz ± 10 ppm) | Stabilité de la molécule : une molécule fragile garde peu d'intensité au précurseur. |
| Ratio isotopique M+1/M et M+2/M | pics à +1,00336 et +1,99705 | Estimation du nombre de C, présence de Cl, Br ou S. |
| **Pertes de neutres** | `precursor_mz − mz_i` pour chaque pic | Les pertes de neutres sont **plus diagnostiques que les fragments eux-mêmes** : H2O (18,011), NH3 (17,027), CO (27,995), CO2 (43,990), CH2O2 (46,005), glucose/hexose (162,053), rhamnose (146,058), pentose (132,042), acide glucuronique (176,032), CH3 (15,023 en mode radicalaire), HCl (35,977)… |
| Fragments diagnostiques | présence de m/z clés (ex. 91,054 tropylium, 105,034 benzoyle, 147,044, 85,029…) | Signatures de sous-structures. |
| Spectre binné | vecteur de taille fixe (ex. 0–1 000 Da, pas de 0,01 ou 0,1 Da), avec intensités en √ ou log | Entrée standard pour un MLP ou un CNN, et pour le cosinus. |
| Formules de fragments | sous-formules de la formule candidate qui expliquent chaque pic (comme dans SIRIUS ou MIST) | Features très puissantes quand une formule candidate est disponible. |

**Prétraitement recommandé pour les pics** : retirer les pics de moins de 0,1–1 % d'intensité, garder les top-k pics (k ≈ 50–100), retirer les pics au-delà de `precursor_mz + 0.5` (en les conservant à part pour les isotopes), appliquer une transformation √ ou log aux intensités, puis renormaliser.

### 7.3 Niveau molécule (agrégation des spectres d'une même `molecule_id`)

Le test fournit **plusieurs spectres par molécule** (CE différentes et parfois les deux polarités). Il faut les combiner :
- **fusionner** les spectres de même polarité (union des pics, max ou moyenne des intensités) ;
- ou prédire **par spectre**, puis **agréger les scores des candidats** (moyenne ou max) ;
- ajouter les features `has_pos`, `has_neg`, `n_spectra` et les adduits disponibles. La présence d'un adduit formiate ou d'un `[M-H]-` fort suggère des groupes acides ou phénoliques.

### 7.4 Côté cible (représentation de la structure)

Pour entraîner un modèle **spectre → structure**, on transforme les SMILES du train en :
- **empreintes moléculaires** : Morgan/ECFP (r=2, 2 048 bits), MACCS (166 bits), fingerprint de sous-structures. C'est l'approche CSI:FingerID/MIST : on prédit la fingerprint, puis on classe des candidats par similarité de Tanimoto ;
- **descripteurs** (formule, nombre de N, de O, d'anneaux, DBE) comme tâches auxiliaires ;
- **SMILES/SELFIES** pour une approche générative de novo (MassGenie, MSNovelist…).

---

## 8. Stratégie de validation (crucial)

- **Splitter par molécule** (`inchikey14`) et **jamais par spectre** : sinon les spectres d'une même molécule fuient du train vers la validation.
- Mieux encore, **splitter par squelette (scaffold Murcko) ou par similarité de Tanimoto**, car les molécules de test sont nouvelles.
- **Validation « proche du test »** : utiliser surtout `enveda-180`, restreint aux 7 adduits du test, à la plage 245–460 de `precursor_mz` et aux CE 20/40/60/fusion, avec un split par scaffold. Le notebook `01_EDA.ipynb` montre que le test se comporte spectralement comme enveda-180 : courbes d'énergie superposées, pertes de neutres et similarité au plus proche voisin. Garder `enveda-np-examples` comme contrôle de robustesse sur des produits naturels.
- Implémenter **exactement la métrique** : canonicalisation des tautomères RDKit, InChIKey14, MRR@25.

---

## 9. Code de départ

```python
import polars as pl, pyarrow.parquet as pq

DATA = "enveda-CASMI26-molecule-id-mass-spectra (1)/"
SCALAR = ["ingest_lib","inchikey14","normalized_smiles","molecular_formula","ionization_mode",
          "instrument_type","adduct","precursor_mz","precursor_error_ppm","num_peaks",
          "collision_energy_orig_units"]
meta = pl.read_parquet(DATA + "train.parquet", columns=SCALAR)   # léger : sans les listes de pics

ADDUCT_SHIFT = {"[M+H]+": 1.007276, "[M-H]-": -1.007276, "[M+Na]+": 22.989218,
                "[M+NH4]+": 18.033823, "[M+K]+": 38.963158, "[M+Cl]-": 34.969402,
                "[M+CH2O2-H]-": 44.998201}

def base_features(df: pl.DataFrame) -> pl.DataFrame:
    return (df.filter(pl.col("adduct").is_in(list(ADDUCT_SHIFT)))
              .with_columns((pl.col("precursor_mz") - pl.col("adduct").replace_strict(ADDUCT_SHIFT)).alias("neutral_mass"))
              .with_columns((pl.col("neutral_mass") - pl.col("neutral_mass").floor()).alias("mass_defect"),
                            (pl.col("neutral_mass").round(0).cast(pl.Int64) % 2).alias("odd_nominal_mass")))

# Lecture des spectres par morceaux (row groups) pour ne pas saturer la RAM
pf = pq.ParquetFile(DATA + "train.parquet")
for i in range(pf.num_row_groups):
    chunk = pl.from_arrow(pf.read_row_group(i))
    ...
```

---

## 10. Résumé

1. **Une ligne correspond à un spectre** ; la cible est la **structure 2D** (InChIKey14), évaluée en **MRR@25**.
2. Les features d'entrée sont peu nombreuses (12 colonnes). L'essentiel de l'information est dans **`precursor_mz` + `adduct`** (qui donnent la masse et la formule) et dans **la liste des pics** (qui donne la structure).
3. Le train est **hétérogène** : 11 sources, 83 noms d'instruments, 4 unités de CE et 121 adduits. Le test est **homogène** : timsTOF, eV, 7 adduits, masse 245–439 Da.
4. Les sources les plus proches du test sont **`enveda-180`** (même plateforme, mais molécules synthétiques azotées et halogénées) et **`enveda-np-examples`** (produits naturels).
5. Les spectres de test sont **denses et bruités** (230 pics en médiane) : le **débruitage** et les **features robustes** (pertes de neutres, entropie, isotopes) seront déterminants.
6. Les molécules de test ne sont **probablement pas dans le train** : il faut un **générateur ou une base de candidats** et un **modèle de scoring** (prédiction de fingerprint ou similarité apprise).

Sources : [Page Kaggle de la compétition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) · [Description de la métrique (repo casmi-2026)](https://github.com/Isaackjoshua/casmi-2026) · [Annonce Enveda CASMI 2026](https://www.webull.com/news/15582421858583552)
