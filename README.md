# 🤖 Reinforcement Learning Project - Modular Framework

Ce projet est une bibliothèque complète d'apprentissage par renforcement (RL) implémentée "from scratch". Il regroupe les principaux algorithmes classiques (Programmation Dynamique, Monte Carlo, Temporal Difference, Planning) et une variété d'environnements de test, incluant des environnements "secrets" pour l'évaluation de la robustesse des agents.

---

## 🎯 Objectifs du Projet

- **Modularité** : Séparation stricte entre les algorithmes, les environnements et les politiques.
- **Extensibilité** : Facilité d'ajout de nouveaux agents ou mondes grâce à des interfaces de base (`BaseAgent`, `BaseEnvironment`).
- **Expérimentation** : Outils pour comparer les performances, étudier la convergence et l'impact des hyperparamètres ($\alpha, \gamma, \epsilon$).
- **Visualisation** : Rendu console et graphique des performances des agents.

---

## 📂 Architecture du Projet

```text
rl-project/
├── rl/
│   ├── algorithms/          # Implémentations des algorithmes RL
│   │   ├── dynamic_programming/  # Policy/Value Iteration
│   │   ├── monte_carlo/         # ES, On-policy, Off-policy
│   │   ├── temporal_difference/ # SARSA, Q-Learning, Expected SARSA
│   │   └── planning/            # Dyna-Q, Dyna-Q+
│   ├── environments/        # Mondes RL (Standard & Secrets)
│   ├── policies/            # Stratégies d'action (Greedy, Epsilon-Greedy)
│   ├── utils/               # Metrics, Logger, Visualisation, Sérialisation
│   └── experiments/         # Scripts pour lancer des tests massifs
├── demo/                    # Scripts interactifs et visualisation
├── saved_models/            # Sauvegarde des politiques entraînées (.pkl)
├── reports/                 # Graphiques et résultats d'expériences
├── main.py                  # Point d'entrée principal
└── requirements.txt         # Dépendances Python
```

---

## 🧠 Algorithmes Implémentés

### 🔹 Programmation Dynamique
*Utilisés quand le modèle (P, R) est connu.*
- **Policy Iteration** : Évaluation et amélioration itérative de la politique.
- **Value Iteration** : Convergence directe vers la fonction de valeur optimale.

### 🔹 Méthodes Monte Carlo
*Apprentissage par épisodes complets.*
- **Monte Carlo ES** (Exploring Starts) : Garantie d'exploration de tous les états.
- **On-policy First-Visit MC** : Apprentissage direct sur la politique cible.
- **Off-policy MC** : Utilisation de l'importance sampling pour apprendre une politique cible via une politique de comportement.

### 🔹 Temporal Difference (TD) Learning
*Apprentissage en ligne (step-by-step).*
- **SARSA** : On-policy, plus sûr pendant l'apprentissage.
- **Q-Learning** : Off-policy, converge vers la politique optimale.
- **Expected SARSA** : Utilise l'espérance mathématique pour réduire la variance.

### 🔹 Planning
- **Dyna-Q** : Combine apprentissage réel et simulations internes (modèle).
- **Dyna-Q+** : Intègre un bonus de curiosité pour découvrir de nouvelles opportunités.

---

## 🌍 Environnements

| Environnement | Description | Intérêt |
| :--- | :--- | :--- |
| **Line World** | Monde 1D simple | Validation des bases. |
| **Grid World** | Monde 2D | Comparaison classique des perfs. |
| **Two-round RPS** | Chifoumi en 2 tours | Dépendance temporelle & stratégie adverse. |
| **Monty Hall (L1)** | Problème des 3 portes | Apprentissage de stratégies contre-intuitives. |
| **Monty Hall (L2)** | 5 portes, multi-étapes | Test de scalabilité. |
| **Secret Envs** | Envs (0-3) inconnus | Test de robustesse et généralisation. |

---

## 🚀 Installation & Utilisation

### Installation
```bash
pip install -r requirements.txt
```

### 🎮 Mode Manuel (Jouer soi-même)
Testez les règles d'un environnement :
```bash
python3 demo/play_manual.py
```

### 🏗️ Entraînement & Expérimentation
Lancez un entraînement par défaut via `main.py` :
```bash
python3 main.py
```

### 🕵️ Tester les Environnements Secrets
Utilisez le script dédié pour tester les environnements fournis par l'enseignant :
```bash
python3 demo/test_secret_envs.py [0-1-2-3]
```

### 📺 Rejouer une Politique Sauvegardée
```bash
python3 demo/replay_policy.py gridworld saved_models/q_values/qlearning_gridworld.pkl
```

---

## 📊 Visualisation
Le framework inclut des outils pour générer :
- Les **Courbes d'apprentissage** (Reward cumulé par épisode).
- Les **Heatmaps de valeur** (V-tables pour les Grids).
- Les **Logs détaillés** dans le dossier `logs/`.

---

## 🛠️ Comment ajouter un composant ?

### Nouvel Environnement
Héritez de `BaseEnvironment` dans `rl/environments/base_env.py` et implémentez `reset()`, `step()`, `get_actions()`, `get_states()` et `render()`.

### Nouvel Algorithme
Héritez de `BaseAgent` dans `rl/algorithms/base_agent.py` et implémentez `train()` et `act()`.

---

> **Note** : Ce projet a été conçu pour être le plus lisible possible afin de faciliter la rédaction du rapport technique final.
