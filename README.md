# Analog Finder (GeoCongo AI)

Outil ouvert qui classe des sites terrestres selon leur ressemblance à un site cible lunaire ou
martien (NASA Space Apps Challenge 2026, defi "Terrestrial Analogs").

## Principe
8 indicateurs, 4 criteres (topographie, geologie, environnement, ressources). Pour chaque
indicateur : distance de Wasserstein entre la distribution de la cible et celle du site, normalisee
par la distance maximale du panel. Aucun modele entraine dans le score.
