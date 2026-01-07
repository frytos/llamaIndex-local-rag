# 🔐 GitHub Authentication pour Runpod (Repo Privé)

Ton repo est **privé**. Pour le cloner dans Runpod, tu dois configurer l'authentification.

---

## 🎯 Solution Recommandée: Personal Access Token (PAT)

### Étape 1: Créer un Token GitHub

**Via Web:**
1. Va sur https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Configure:
   ```
   Note: Runpod RAG Pipeline
   Expiration: 90 days (ou plus)
   Scopes: ✅ repo (tous les sous-scopes)
   ```
4. Click **"Generate token"**
5. **COPIE LE TOKEN** (tu ne pourras plus le revoir!)
   - Format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**Via CLI (plus rapide):**
```bash
gh auth token
```

Ou créer un nouveau:
```bash
gh auth login --scopes repo
```

---

### Étape 2: Configurer Runpod

#### Option A: Environment Variable (Recommandé ⭐)

Dans Runpod UI, ajoute cette **Environment Variable**:

```
Key: GH_TOKEN
Value: ghp_ton_token_ici
```

Puis utilise cette **Docker Command**:

```bash
bash -c "apt-get update -qq && apt-get install -y git && git clone https://\${GH_TOKEN}@github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"
```

#### Option B: Hardcode dans la commande (Moins sûr)

Remplace `TON_TOKEN` par ton vrai token:

```bash
bash -c "apt-get update -qq && apt-get install -y git && git clone https://ghp_TON_TOKEN@github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"
```

⚠️ **Attention:** Le token sera visible dans les logs Runpod!

---

## 🔓 Alternative: Rendre le Repo Public (Temporairement)

Si tu veux tester rapidement sans token:

```bash
# Rendre public
gh repo edit llamaIndex-local-rag --visibility public

# Test dans Runpod avec la commande normale
# (pas besoin de token)

# Remettre privé après test
gh repo edit llamaIndex-local-rag --visibility private
```

---

## 🧪 Tester l'Authentification

### Localement (vérifier le token):

```bash
# Set ton token
export GH_TOKEN=ghp_ton_token_ici

# Test le clone
git clone https://${GH_TOKEN}@github.com/frytos/llamaIndex-local-rag.git /tmp/test-clone

# Si ça marche, supprimer:
rm -rf /tmp/test-clone
```

### Dans Runpod (vérifier après startup):

```bash
# Se connecter au pod
cd /workspace/rag-pipeline

# Vérifier que le clone a réussi
ls -la

# Output attendu: tous les fichiers du repo
```

---

## 🔒 Sécurité du Token

### Bonnes Pratiques:

1. ✅ **Utilise l'Environment Variable** dans Runpod (pas hardcodé)
2. ✅ **Scope minimal:** Seulement `repo` (pas `admin`, `delete`, etc.)
3. ✅ **Expiration courte:** 90 jours ou moins
4. ✅ **Révoque après usage:** Si tu testes seulement, révoque le token après
5. ✅ **Ne commit jamais le token** dans le code!

### Révoquer un Token:

**Via Web:**
https://github.com/settings/tokens → Click "Delete"

**Via CLI:**
```bash
gh auth token  # Copy le token
gh api -X DELETE /applications/{client_id}/tokens/{token}
```

---

## 🐛 Troubleshooting

### Error: "Authentication failed"

**Cause:** Token invalide ou expiré

**Solution:**
```bash
# Créer un nouveau token
gh auth token

# Ou login de nouveau
gh auth login --scopes repo
```

### Error: "Repository not found"

**Cause:** Token sans le scope `repo`

**Solution:** Recrée un token avec le scope `repo` coché

### Error: "fatal: could not read Username"

**Cause:** Format d'URL incorrect

**Solution:** Vérifie le format:
```bash
# ✅ Correct
https://${GH_TOKEN}@github.com/frytos/llamaIndex-local-rag.git

# ❌ Incorrect
https://github.com/frytos/llamaIndex-local-rag.git  # manque token
```

---

## 📋 Checklist Avant Runpod Deploy

- [ ] Token GitHub créé (`gh auth token`)
- [ ] Token copié (commence par `ghp_`)
- [ ] Environment Variable `GH_TOKEN` ajoutée dans Runpod UI
- [ ] Docker Command mise à jour avec `\${GH_TOKEN}@`
- [ ] Test local du clone avec token (optionnel)
- [ ] Deploy Runpod!

---

## 🚀 TL;DR - Quick Start

```bash
# 1. Créer token
gh auth token  # Copie le résultat

# 2. Dans Runpod UI:
#    Environment Variables → Add:
#      GH_TOKEN = ton_token

# 3. Docker Command:
bash -c "apt-get update -qq && apt-get install -y git && git clone https://\${GH_TOKEN}@github.com/frytos/llamaIndex-local-rag.git /workspace/rag-pipeline && cd /workspace/rag-pipeline && chmod +x scripts/runpod_startup.sh && SETUP_POSTGRES=1 DOWNLOAD_MODELS=1 bash scripts/runpod_startup.sh"

# 4. Deploy!
```

**C'est tout!** 🎉

---

## 📚 Références

- GitHub Personal Access Tokens: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
- GitHub CLI: https://cli.github.com/manual/
- Git Credential Helper: https://git-scm.com/docs/gitcredentials
