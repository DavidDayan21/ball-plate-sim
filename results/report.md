# Bilan de la campagne de mesures du robot équilibriste simulé

Chaîne simulée : MuJoCo (robot à 3 bras, rampes moteurs) + caméra 160×120 + OpenCV + PID.


## 0. Du modèle simple à la chaîne complète (scénario de référence, mêmes réglages)

| Niveau | Tombée | Retour 5 % | Erreur cercle | Agitation θ | Décollée |
|---|---|---|---|---|---|
| Maison : plateau en angles | non | 2.76 s | 3.34 mm | 0.55° | - |
| Maison : robot 3 bras | non | 2.72 s | 4.35 mm | 0.57° | - |
| MuJoCo : plateau en angles | non | 3.98 s | 3.01 mm | 0.55° | 75.2 % |
| MuJoCo : robot 3 bras | non | 2.68 s | 5.46 mm | 0.58° | 0.0 % |
| MuJoCo : robot 3 bras + caméra | non | 2.78 s | 4.61 mm | 0.12° | 0.0 % |


## 1. Limites théoriques et étalonnage du contact

| Grandeur | Théorie | MuJoCo |
|---|---|---|
| Frottement minimal pour rouler à 10° | 0.071 | - |
| Accélération à 10°, frottement 1,3×min (roule) | 1.022 m/s² | 1.022 m/s² |
| Accélération à 10°, frottement 0,7×min (glisse) | 1.227 m/s² | 1.234 m/s² |
| Restitution, lâcher de 30,5 cm sur plan fixe (règle ITTF) | 0,85 à 0,91 | 0.874 |


## 2. Balle lancée depuis x = -10 cm vers le bord opposé (consigne : centre)

| Inclinaison max | Vitesse max rattrapée | Limite physique √(2·c·g·sin θmax·d) |
|---|---|---|
| ±10° | 0.5 m/s | 0.71 m/s |
| ±20° | 0.6 m/s | 1.00 m/s |


## 3. Poussée sur la balle au repos au centre (t = 1 s)

| Choc max encaissé (Δv) | Limite physique (d = 15 cm, ±10°) |
|---|---|
| 0.4 m/s | 0.55 m/s |


## 4. Robustesse aux paramètres physiques (contrôleur inchangé, réglé pour la balle de ping-pong)

| Variante | Tombée | Retour 5 % | Erreur cercle | Décollée |
|---|---|---|---|---|
| Nominal (ping-pong, μ = 1) | non | 2.78 s | 4.61 mm | 0.0 % |
| Bille pleine (k = 2/5) | non | 2.90 s | 7.11 mm | 0.0 % |
| Balle 10× plus lourde (27 g) | non | 2.78 s | 4.61 mm | 0.0 % |
| Frottement μ = 0,2 | non | 2.78 s | 4.45 mm | 0.0 % |
| Frottement μ = 0,02 | non | 2.84 s | 4.47 mm | 0.0 % |
| Plateau 2× plus lourd (0,5 kg) | non | 2.78 s | 4.56 mm | 0.0 % |


## 5. Latence de la caméra : sans et avec compensation par prédiction

| Latence | Compensation | Tombée | Retour 5 % | Erreur cercle |
|---|---|---|---|---|
| 20 ms | sans | non | 2.78 s | 4.61 mm |
| 20 ms | avec | non | 2.82 s | 1.47 mm |
| 60 ms | sans | non | 2.68 s | 10.56 mm |
| 60 ms | avec | non | 2.82 s | 1.66 mm |
| 100 ms | sans | non | 2.62 s | 16.66 mm |
| 100 ms | avec | non | 3.72 s | 1.94 mm |


## 6. Trajectoire carrée (10 cm de côté, 5 cm/s), latence 20 ms

| Compensation | Tombée | Erreur (2e tour) |
|---|---|---|
| sans | non | 9.81 mm |
| avec | non | 9.34 mm |

