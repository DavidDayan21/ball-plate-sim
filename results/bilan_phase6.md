# Bilan de la phase 6 : robustesse du robot équilibriste simulé

Chaîne simulée : MuJoCo (robot à 3 bras, rampes moteurs) + caméra 160×120 + OpenCV + PID.


## 1. Limites théoriques

| Grandeur | Théorie | MuJoCo |
|---|---|---|
| Frottement minimal pour rouler à 10° | 0.071 | - |
| Accélération à 10°, frottement 1,3×min (roule) | 1.022 m/s² | 1.022 m/s² |
| Accélération à 10°, frottement 0,7×min (glisse) | 1.227 m/s² | 1.234 m/s² |


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

| Variante | Tombée | Retour 5 % | Erreur cercle | Sans contact |
|---|---|---|---|---|
| Nominal (ping-pong, μ = 1) | non | 2.78 s | 4.41 mm | 0.1 % |
| Bille pleine (k = 2/5) | non | 2.90 s | 6.77 mm | 0.2 % |
| Balle 10× plus lourde (27 g) | non | 2.78 s | 4.44 mm | 0.1 % |
| Frottement μ = 0,2 | non | 2.78 s | 4.50 mm | 1.6 % |
| Frottement μ = 0,02 | non | 2.84 s | 4.44 mm | 2.2 % |
| Plateau 2× plus lourd (0,5 kg) | non | 2.80 s | 4.64 mm | 1.4 % |


## 5. Latence de la caméra : sans et avec compensation par prédiction

| Latence | Compensation | Tombée | Retour 5 % | Erreur cercle |
|---|---|---|---|---|
| 20 ms | sans | non | 2.78 s | 4.41 mm |
| 20 ms | avec | non | 2.82 s | 1.45 mm |
| 60 ms | sans | non | 2.68 s | 10.51 mm |
| 60 ms | avec | non | 3.68 s | 1.63 mm |
| 100 ms | sans | non | 2.62 s | 16.74 mm |
| 100 ms | avec | non | 3.98 s | 1.99 mm |


## 6. Trajectoire carrée (10 cm de côté, 5 cm/s), latence 20 ms

| Compensation | Tombée | Erreur (2e tour) |
|---|---|---|
| sans | non | 10.06 mm |
| avec | non | 9.58 mm |

