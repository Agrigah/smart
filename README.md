# ⌨️ Smart Keyboard V1

Moteur de suggestion du mot suivant, comme sur un clavier de téléphone, basé sur un **modèle de langage n-gramme** (unigramme, bigramme, trigramme). Projet du Lab 02, avec une interface web Streamlit.

🔗 **Démo en ligne :** https://mdrgbmdehvubq9sshmwfcy.streamlit.app

---

## Objectif

Prédire le mot suivant à partir de ce qui a déjà été tapé, puis mesurer la qualité du modèle avec la **perplexité** et la **précision top-3**.

Le modèle estime P(mot | contexte) à partir de comptages, avec un lissage add-k pour que les séquences jamais vues n'aient pas une probabilité nulle :

```
P(w | c) = (count(c, w) + k) / (count(c) + k · V)
```

où `V` est la taille du vocabulaire et `k` la constante de lissage.

## Ce que le projet contient

- **Modèles n-grammes** : unigramme, bigramme et trigramme, avec les symboles `<s>` (début), `</s>` (fin) et `<unk>` (mot inconnu).
- **Lissage add-k** pour éviter les probabilités nulles.
- **Back-off** : on utilise le trigramme, puis le bigramme, puis l'unigramme si le contexte n'a jamais été vu.
- **Interpolation linéaire** : `0.6·P₃ + 0.3·P₂ + 0.1·P₁`.
- **Évaluation** sur des documents de test : perplexité et précision top-3.
- **Découpage par document** (et non par phrase) pour éviter la fuite de données.
- **Interface Streamlit** avec suggestions en direct.

## Structure du dépôt

```
.
├── app.py              # Interface Streamlit
├── requirements.txt    # Dépendances Python
├── data/
│   ├── corpus.csv      # 1 200 phrases synthétiques (60 documents)
│   └── make_data.py    # Génère le corpus (graine fixe)
├── src/
│   ├── pipeline.py     # Modèles n-grammes, back-off, interpolation, métriques
│   └── evaluate.py     # Évaluation complète et génération des résultats
└── results/
    ├── metrics.json    # Perplexité et top-3 de chaque modèle
    ├── failures.txt    # Analyse des erreurs
    ├── run_log.txt     # Journal de l'évaluation
    └── ngram_results.png
```

## Installation et lancement

Prérequis : Python 3.10 ou plus récent.

```bash
# 1. Récupérer le projet
git clone https://github.com/Agrigah/smart.git
cd smart

# 2. (Optionnel) créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

### Régénérer les données et refaire l'évaluation (optionnel)

Le corpus et les résultats sont déjà fournis. Pour les recalculer :

```bash
python data/make_data.py       # régénère data/corpus.csv
python -m src.evaluate         # recalcule results/
```

## Résultats

Évaluation sur les 20 % de documents réservés au test (240 phrases, 2 070 tokens) :

| Modèle | Perplexité | Précision top-3 |
|---|---:|---:|
| Unigramme (k = 1) | 24,96 | 48,0 % |
| Bigramme (k = 1) | 6,43 | 60,9 % |
| Trigramme (k = 1) | 8,30 | 64,1 % |
| **Trigramme (k = 0,1)** | **5,06** | 64,1 % |
| Trigramme sans lissage (k = 0) | ∞ | 64,1 % |
| Interpolé 3/2/1 (k = 0,1) | 5,15 | **64,7 %** |

Ce qu'on en retient :

- Plus le contexte est long, meilleur est le modèle, jusqu'à ce que les données manquent.
- Sans lissage (k = 0), 4,6 % des tokens de test reçoivent une probabilité nulle, donc la perplexité est infinie.
- Trop de lissage (k = 1 ou plus) dégrade aussi la perplexité : le meilleur k se situe autour de 0,01 à 0,1.
- Les erreurs viennent surtout de contextes rares ou jamais vus à l'entraînement, par exemple « the » seul en début de phrase.

## Limites

- Le corpus est **synthétique** (61 mots seulement, grammaire fixe) : les probabilités ne reflètent pas de vraies statistiques de frappe.
- Le modèle n'a qu'une mémoire de deux mots (trigramme) et ne comprend pas le sens des phrases.
- Les mots absents du vocabulaire sont traités comme `<unk>`.

## Technologies

Python · Streamlit · pandas · matplotlib
