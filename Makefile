-include .env
export

COMPOSE   ?= docker compose
API       ?= http://localhost:8080
TRACKING  ?= http://localhost:5000
BUNDLE    ?= models
ADMIN_TOKEN ?= admin-dev-token
N         ?= 3
VM_USER   ?= root
TF_DIR    ?= infra/terraform
HOST      ?= http://localhost:8080
USERS     ?= 50
RATE      ?= 5
DURATION  ?= 3m
PROD      = docker compose -f docker-compose.prod.yml

.PHONY: help up down logs ps scale user users usage register canary promote drift test \
        loadtest loadtest-drift tf-init tf-apply tf-destroy tunnel deploy

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------- Niveau 0 : local ----------
up: ## Lance toute la stack en local
	$(COMPOSE) up -d --build

down: ## Arrête la stack
	$(COMPOSE) down

logs: ## Logs de tous les services
	$(COMPOSE) logs -f --tail=100

ps: ## État des conteneurs
	$(COMPOSE) ps

scale: ## Scaling horizontal de l'API : make scale N=3
	$(COMPOSE) up -d --no-recreate --scale api=$(N)
	$(COMPOSE) exec nginx nginx -s reload

# ---------- Utilisateurs / usage ----------
user: ## Crée un utilisateur et affiche sa clé : make user NAME=alice
	@curl -s -X POST $(API)/admin/users -H "X-Admin-Token: $(ADMIN_TOKEN)" -H "Content-Type: application/json" -d '{"name":"$(NAME)"}'; echo

users: ## Liste les utilisateurs
	@curl -s $(API)/admin/users -H "X-Admin-Token: $(ADMIN_TOKEN)"; echo

usage: ## Usage par utilisateur sur 24h
	@curl -s "$(API)/admin/usage?hours=24" -H "X-Admin-Token: $(ADMIN_TOKEN)"; echo

# ---------- MLflow ----------
register: ## Enregistre BUNDLE dans MLflow avec l'alias production
	python training/register.py --bundle $(BUNDLE) --tracking-uri $(TRACKING) --alias production

canary: ## Enregistre BUNDLE avec l'alias canary : make canary BUNDLE=models_v2
	python training/register.py --bundle $(BUNDLE) --tracking-uri $(TRACKING) --alias canary

promote: ## Promeut une version en production : make promote V=2
	python training/promote.py --tracking-uri $(TRACKING) --version $(V) --alias production

# ---------- Drift / tests ----------
drift: ## Analyse de drift immédiate
	$(COMPOSE) exec drift python drift_job.py --once

test: ## Tests unitaires de l'API
	cd api && python -m pytest -q

loadtest: ## Test de charge : make loadtest HOST=http://IP:8080 TOKEN=sk_...
	API_TOKEN=$(TOKEN) locust -f loadtest/locustfile.py --host $(HOST) --headless -u $(USERS) -r $(RATE) -t $(DURATION) --html loadtest/report.html

loadtest-drift: ## Envoie des images dégradées pour déclencher le drift
	API_TOKEN=$(TOKEN) DRIFT_MODE=1 locust -f loadtest/locustfile.py --host $(HOST) --headless -u 10 -r 2 -t 2m --html loadtest/report_drift.html

# ---------- Niveau 1/2 : infra + déploiement ----------
tf-init: ## terraform init (TF_DIR=infra/terraform-aws pour AWS)
	cd $(TF_DIR) && terraform init

tf-apply: ## Crée la VM (Scaleway par défaut, AWS avec TF_DIR=infra/terraform-aws)
	cd $(TF_DIR) && terraform apply

tf-destroy: ## Détruit la VM
	cd $(TF_DIR) && terraform destroy

tunnel: ## Tunnel SSH vers MLflow / Grafana / Prometheus de la VM : make tunnel IP=x.x.x.x
	ssh -N -L 5000:localhost:5000 -L 3000:localhost:3000 -L 9090:localhost:9090 $(VM_USER)@$(IP)

deploy: ## Déploiement manuel sans CI (build sur la VM) : make deploy IP=x.x.x.x
	rsync -az --delete --exclude .git --exclude data --exclude '.terraform*' --exclude '*.tfstate*' --exclude .venv ./ $(VM_USER)@$(IP):/opt/foodvision/
	ssh $(VM_USER)@$(IP) "cd /opt/foodvision && docker compose up -d --build"
