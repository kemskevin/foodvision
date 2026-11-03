# 🍽️ FoodVision — Projet MLOps

Classification d'images de plats (**Food-101**, 101 classes) avec un **EfficientNet-B0 fine-tuné**,
exporté en **ONNX** et **quantisé en INT8**, servi par une API **FastAPI** sécurisée par token,
déployé automatiquement sur une VM **Scaleway** créée avec **Terraform**.

## Architecture

```
                     ┌────────────────┐
  Navigateur ──:80──►│     Nginx      │──► UI Streamlit
                     │                │──► /reports/ (rapports de drift)
  Client API ─:8080─►│ split_clients  │
                     └───┬────────┬───┘
                    90 % │        │ 10 %   (canary deployment)
              ┌──────────▼──┐  ┌──▼──────────┐
              │ api × N     │  │ api_canary  │   FastAPI + ONNX Runtime
              │ alias       │  │ alias       │   auth X-API-Key
              │ production  │  │ canary      │   rechargement auto depuis MLflow
              └──────┬──────┘  └──────┬──────┘
     ┌───────────────┼────────────────┼──────────────────┐
     ▼               ▼                ▼                  ▼
 PostgreSQL       MLflow          Prometheus ──► Grafana     Drift (Evidently)
 users, tokens,   tracking +                                lit usage_logs,
 usage_logs       model registry                            compare à reference.csv
```

## Correspondance avec la grille

| Niveau | Exigence | Implémentation |
|---|---|---|
| 0 | Une commande, services conteneurisés | `make up` (= `docker compose up -d --build`), 9 services |
| 1 | Déploiement auto sur VM | GitHub Actions : tests → build images GHCR → déploiement SSH (`.github/workflows/cicd.yml`), ou `make deploy IP=…` |
| 1 | Token dans le header | `X-API-Key`, tokens hachés SHA-256 en base, révocables (`api/app/auth.py`) |
| 1 | Usage par utilisateur | table `usage_logs`, `GET /admin/usage`, `GET /me/usage` |
| 1 | Test de charge | Locust (`loadtest/locustfile.py`), rapport HTML |
| 2 | Terraform | `infra/terraform` : VM, IP, security group, clé SSH, cloud-init Docker |
| 2 | Interface utilisateur | Streamlit (`ui/app.py`) |
| 2 | Data drift | Evidently (`drift/drift_job.py`) : rapports HTML + métriques Prometheus |
| 2 | MLflow | tracking + model registry, alias `production` / `canary`, rechargement à chaud |
| 3 | Quantization | ONNX Runtime static INT8 (QDQ), benchmark FP32 vs INT8 loggé dans MLflow |
| 3 | Canary deployment | Nginx `split_clients`, `CANARY_PERCENT` configurable |
| 3 | Scaling horizontal | `deploy.replicas` / `make scale N=3`, Prometheus découvre les réplicas par DNS |
| + | Monitoring | Prometheus + dashboard Grafana provisionné |

---

## 1. Entraîner le modèle (Colab, ~40 min)

Sur Colab, choisir un runtime GPU T4, puis :

```python
!git clone https://github.com/<toi>/foodvision.git
%cd foodvision/training
!pip install -q onnx onnxruntime
!python train.py --dataset food101 --arch efficientnet_b0 --epochs 3 --out artifacts
!zip -r artifacts.zip artifacts
from google.colab import files; files.download("artifacts.zip")
```

Pour un premier essai rapide, utiliser `--max-train-samples 15000 --epochs 2`, ou `--dataset pets`.

Dézipper ensuite dans `models/` à la racine du repo. On doit obtenir `models/model_int8.onnx`,
`models/labels.json`, `models/reference.csv`, `models/samples/`, etc.

> Si la précision INT8 (`ref_top1_onnx_int8` dans `metrics.json`) perd plus de 2 points par rapport au FP32,
> relancer avec `--calib-method percentile`. C'est un bon point à discuter à l'oral.

## 2. Lancer en local (niveau 0)

```bash
cp .env.example .env        # puis modifier les mots de passe
make up                     # ou : docker compose up -d --build
```

| Service | URL |
|---|---|
| UI | http://localhost |
| API (docs Swagger) | http://localhost:8080/docs |
| MLflow | http://localhost:5000 |
| Grafana (admin / GRAFANA_PASSWORD) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Rapports de drift | http://localhost/reports/ |

Sans modèle enregistré dans MLflow, l'API utilise le bundle local `models/`.

```bash
pip install -r training/requirements-register.txt
make register               # bundle models/ -> MLflow, alias production (v1)
make user NAME=alice        # -> {"api_key": "sk_..."}  à conserver
curl -X POST localhost:8080/predict -H "X-API-Key: sk_..." -F "file=@models/samples/00_pizza.jpg"
make usage                  # usage par utilisateur
```

Sous Windows, utiliser WSL2 (Docker Desktop + Ubuntu) pour `make`, `rsync` et `ssh`.

## 3. Infrastructure Scaleway (niveau 2)

1. Créer un compte Scaleway, puis une clé API (IAM → API keys).
2. Exporter les identifiants et lancer Terraform :

```bash
export SCW_ACCESS_KEY=... SCW_SECRET_KEY=... SCW_DEFAULT_PROJECT_ID=... SCW_DEFAULT_ORGANIZATION_ID=...
make tf-init
make tf-apply               # affiche public_ip, api_url, ui_url
```

Une DEV1-M coûte environ 0,02 €/h. Faire `make tf-destroy` après la soutenance.
cloud-init installe Docker automatiquement, compter 2 à 3 minutes après la création.

## 4. Déploiement continu (niveau 1)

Dans GitHub → Settings → Secrets and variables → Actions, créer les secrets suivants :

| Secret | Valeur |
|---|---|
| `VM_HOST` | IP publique (sortie Terraform) |
| `VM_SSH_KEY` | clé **privée** correspondant à la clé publique donnée à Terraform |
| `POSTGRES_PASSWORD`, `ADMIN_TOKEN`, `GRAFANA_PASSWORD` | mots de passe de prod |

Les variables `API_REPLICAS` (défaut 2) et `CANARY_PERCENT` (défaut 10) sont optionnelles.

Chaque push sur `main` déclenche la chaîne : tests → build et push des 3 images sur GHCR → copie des fichiers → `docker compose pull && up -d` sur la VM → health check.

Ensuite, enregistrer le modèle dans le MLflow de prod via un tunnel SSH (MLflow n'est pas exposé publiquement) :

```bash
make tunnel IP=<ip>                              # terminal 1
make register TRACKING=http://localhost:5000     # terminal 2 -> les API chargent v1 en < 60 s
make user API=http://<ip>:8080 NAME=alice
```

> Pour pouvoir faire `docker compose pull` à la main sur la VM, passer les packages GHCR en public
> (le token de la CI expire à la fin du job).

## 5. Test de charge

```bash
pip install -r loadtest/requirements.txt
make loadtest HOST=http://<ip>:8080 TOKEN=sk_... USERS=50 DURATION=3m
```

Le rapport est écrit dans `loadtest/report.html`. Pour la soutenance, comparer 3 configurations :
1 réplica FP32 (`MODEL_VARIANT=fp32`), 1 réplica INT8, puis 3 réplicas INT8.
Les indicateurs à relever sont le RPS max, la latence p95 et le taux d'erreurs.

## 6. Data drift

Le service `drift` tourne toutes les 5 minutes. Il compare la luminosité, le contraste, la netteté,
la confiance et la distribution des classes prédites au jeu de référence versionné avec le modèle.

Pour la démo :

```bash
make loadtest HOST=... TOKEN=... DURATION=1m     # trafic normal
make drift                                       # pas de drift attendu
make loadtest-drift HOST=... TOKEN=...           # images sombres et floues
make drift                                       # drift détecté
```

Le résultat est visible sur `http://<ip>/reports/latest.html` et dans le dashboard Grafana.

## 7. Canary deployment et mise à jour du modèle

```bash
# entraîner un challenger, par ex. --arch mobilenet_v3_large ou plus d'epochs -> models_v2/
make canary BUNDLE=models_v2      # v2 avec l'alias canary -> 10 % du trafic
# comparer latence et confiance par déploiement dans Grafana, puis :
make promote V=2                  # v2 passe en production, rechargement automatique
```

Le header `X-Model-Version` de chaque réponse indique quelle version a répondu.

## 8. Scaling horizontal

```bash
make scale N=3       # en local
```

En prod, définir la variable GitHub `API_REPLICAS=3` puis relancer la CD.

## Endpoints

| Méthode | Route | Auth |
|---|---|---|
| GET | `/health` | — |
| POST | `/predict` (multipart `file`, `?top_k=5`) | `X-API-Key` |
| GET | `/me/usage` | `X-API-Key` |
| POST | `/admin/users` `{"name": "alice"}` | `X-Admin-Token` |
| GET | `/admin/users` | `X-Admin-Token` |
| DELETE | `/admin/users/{name}` (révocation) | `X-Admin-Token` |
| GET | `/admin/usage?hours=24` | `X-Admin-Token` |
| POST | `/admin/reload` | `X-Admin-Token` |
