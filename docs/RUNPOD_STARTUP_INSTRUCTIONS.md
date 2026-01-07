# 🚀 Runpod Startup Command - Instructions

## Option 1: One-Liner (Recommandé) ⭐

### Dans l'UI Runpod

Quand tu crées ton pod, dans le champ **"Docker Command"**, colle ceci:

```bash
bash -c "apt-get update -qq && apt-get install -y git && rm -rf /workspace/rag-pipeline && git clone https://github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && bash scripts/runpod_startup.sh"
```

### Avec Variables d'Environnement (Avancé)

Si tu veux customiser le comportement, ajoute ces **Environment Variables** dans l'UI:

| Variable | Description | Défaut | Exemple |
|----------|-------------|--------|---------|
| `REPO_URL` | URL du repo Git | https://github.com/frytos/llamaIndex-local-rag.git | Ton fork |
| `SETUP_POSTGRES` | Installer PostgreSQL local | 0 | `1` (pour activer) |
| `DOWNLOAD_MODELS` | Pré-télécharger les modèles | 0 | `1` (pour activer) |
| `RUN_COMMAND` | Commande à exécuter après setup | (none) | `python3 rag_low_level_m1_16gb_verbose.py --help` |
| `KEEP_ALIVE` | Garder le container actif | 0 | `1` (pour debug) |

---

## Option 2: Startup Script dans le Pod

### Étape 1: Upload le script

Après avoir créé ton pod:

```bash
# SSH dans le pod
ssh root@your-pod-ip -p your-port

# Clone le repo
rm -rf /workspace/rag-pipeline && git clone https://github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline
cd /workspace/rag-pipeline

# Rendre le script exécutable
chmod +x scripts/runpod_startup.sh

# Exécuter
bash scripts/runpod_startup.sh
```

### Étape 2: Avec Variables

```bash
# Setup complet avec PostgreSQL et modèles
SETUP_POSTGRES=1 \
DOWNLOAD_MODELS=1 \
bash scripts/runpod_startup.sh
```

---

## Option 3: Configuration Automatique Complète

### Dans Runpod UI - Configuration du Pod

**1. Template:** Runpod PyTorch 2.4.0

**2. GPU:** RTX 4090 (24GB)

**3. Container Disk:** 50 GB

**4. Volume Disk:** 100 GB

**5. Expose Ports:**
```
5432,8000,22
```

**6. Environment Variables:**
```bash
SETUP_POSTGRES=1
DOWNLOAD_MODELS=1
EMBED_BACKEND=torch
N_GPU_LAYERS=99
N_BATCH=512
CTX=16384
PGHOST=localhost
PGUSER=fryt
PGPASSWORD=frytos
DB_NAME=vector_db
PGTABLE=messenger_runpod
HF_HOME=/workspace/huggingface_cache
```

**7. Docker Command:**
```bash
bash -c "apt-get update -qq && apt-get install -y git && rm -rf /workspace/rag-pipeline && git clone https://github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"
```

**8. Deploy!**

---

## 🎯 Ce Que Fait le Startup Script

### Automatiquement:

1. ✅ Affiche les infos GPU (nvidia-smi)
2. ✅ Clone ton repo (ou pull si déjà cloné)
3. ✅ Crée un virtual environment Python
4. ✅ Installe toutes les dépendances (requirements.txt)
5. ✅ Installe PyTorch 2.4.0 avec CUDA 12.4
6. ✅ Charge la configuration (runpod_config.env)
7. ✅ Setup PostgreSQL (si SETUP_POSTGRES=1)
8. ✅ Test le GPU + PyTorch
9. ✅ Pré-télécharge les modèles (si DOWNLOAD_MODELS=1)
10. ✅ Affiche un résumé + commandes utiles

### Temps total: ~2-3 minutes

---

## 📊 Après le Startup

### Vérifier que tout fonctionne:

```bash
# Se connecter au pod (Web Terminal ou SSH)
cd /workspace/rag-pipeline
source .venv/bin/activate

# Tester le GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Tester PostgreSQL (si installé)
psql -h localhost -U fryt -d vector_db -c "SELECT version();"

# Run une query test
python3 rag_low_level_m1_16gb_verbose.py --query-only \
  --query "test query"
```

---

## 🔧 Troubleshooting

### Le script ne démarre pas

**Problème:** `git: command not found`
**Solution:** Le one-liner installe git automatiquement, mais si tu utilises un custom template, assure-toi que git est installé:
```bash
apt-get update && apt-get install -y git
```

### PyTorch n'a pas CUDA

**Problème:** `torch.cuda.is_available() = False`
**Solution:** Vérifie que tu utilises bien le template **PyTorch 2.4.0** avec CUDA support. Runpod a parfois des templates CPU-only par erreur.

### PostgreSQL connection refused

**Problème:** `could not connect to server`
**Solution:**
```bash
# Vérifier si PostgreSQL est démarré
service postgresql status

# Le démarrer si nécessaire
service postgresql start

# Tester la connexion
psql -h localhost -U fryt -d vector_db
```

### Repo déjà existe (erreur git clone)

**Problème:** `fatal: destination path already exists`
**Solution:** Le script détecte automatiquement et fait un `git pull` au lieu de `clone`. Si problème persiste:
```bash
rm -rf /workspace/rag-pipeline
# Puis relance le script
```

---

## 💡 Tips & Best Practices

### 1. Utilise un Volume Persistant

Configure un **Network Volume** dans Runpod pour:
- `/workspace/rag-pipeline` (ton code)
- `/workspace/huggingface_cache` (modèles pré-téléchargés)
- `/var/lib/postgresql` (données PostgreSQL)

**Pourquoi?** Les volumes persistent même si tu stop/restart le pod. Tu ne repays pas le download des modèles!

### 2. Fork le Repo

Au lieu d'utiliser mon repo, fork-le et utilise ton propre URL:
```bash
REPO_URL=https://github.com/TON-USERNAME/llamaIndex-local-rag.git
```

**Pourquoi?** Tu peux pusher tes modifications et le pod les pulera automatiquement.

### 3. Test en Local d'Abord

Avant de mettre le startup command dans Runpod, teste-le en local:
```bash
# Dans ton terminal local
bash scripts/runpod_startup.sh
```

### 4. Monitore les Logs

Pendant le premier startup:
```bash
# Dans le Web Terminal Runpod
tail -f /workspace/rag-pipeline/*.log
```

### 5. Stop/Start Workflow

Pour économiser:
1. **Start:** Le startup script setup tout automatiquement
2. **Work:** Fais tes benchmarks/tests
3. **Stop:** Arrête le pod
4. **Restart:** Le script pull les derniers changements et repart

---

## 🚀 Quick Start - TL;DR

### Méthode Ultra-Rapide:

1. **Runpod UI** → Deploy GPU Pod
2. **Template:** PyTorch 2.4.0
3. **GPU:** RTX 4090
4. **Docker Command:**
```bash
bash -c "apt-get update -qq && apt-get install -y git && rm -rf /workspace/rag-pipeline && git clone https://github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"
```
5. **Deploy!**
6. Attends 2-3 minutes
7. **Connect** → Web Terminal
8. **Run:**
```bash
cd /workspace/rag-pipeline
source .venv/bin/activate
python3 rag_low_level_m1_16gb_verbose.py --query-only --query "test"
```

**C'est tout!** 🎉

---

## 📚 Références

- Script source: `scripts/runpod_startup.sh`
- Config exemple: `runpod_config.env`
- Guide complet: `RUNPOD_DEPLOYMENT_GUIDE.md`
- Runpod Docs: https://docs.runpod.io/

---

## 🆘 Besoin d'Aide?

Si quelque chose ne marche pas:
1. Check les logs: `cat /workspace/rag-pipeline/setup.log`
2. Vérifie GPU: `nvidia-smi`
3. Test PyTorch: `python3 -c "import torch; print(torch.cuda.is_available())"`
4. Demande-moi! 💬
