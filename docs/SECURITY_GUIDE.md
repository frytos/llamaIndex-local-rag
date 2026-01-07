# Security Guide

**Last Updated**: January 2026
**Version**: 1.0.0

## Overview

This guide covers security best practices for deploying and operating the Local RAG Pipeline.

## Quick Security Checklist

- [ ] Never commit `.env` file to version control
- [ ] Use strong, unique database passwords
- [ ] Rotate credentials regularly (every 90 days)
- [ ] Enable Web UI authentication in production
- [ ] Use TLS/SSL for database connections
- [ ] Keep all dependencies updated
- [ ] Run security scans regularly
- [ ] Restrict network access to services
- [ ] Enable audit logging
- [ ] Back up data securely

## Credential Management

### Environment Variables

All sensitive credentials MUST be stored in environment variables, never hardcoded.

**Correct** ✓
```bash
# .env file
PGPASSWORD=my_secure_password_here
```

**Wrong** ✗
```python
# Never do this!
conn = psycopg2.connect(password="hardcoded_password")
```

### Setting Up Credentials

1. **Copy the template**:
   ```bash
   cp config/.env.example .env
   ```

2. **Generate strong passwords**:
   ```bash
   # On macOS/Linux
   openssl rand -base64 32

   # Or use Python
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. **Set all required variables**:
   ```bash
   # Edit .env file
   PGUSER=your_db_user
   PGPASSWORD=your_strong_password_here
   DB_NAME=vector_db
   ```

4. **Verify .env is in .gitignore**:
   ```bash
   grep -q "^\.env$" .gitignore && echo "OK" || echo "WARNING: Add .env to .gitignore!"
   ```

### Password Requirements

- Minimum 16 characters
- Mix of uppercase, lowercase, numbers, symbols
- No dictionary words
- Unique per environment (dev/staging/prod)
- Stored in password manager

### Credential Rotation

Rotate database credentials every 90 days:

```bash
# 1. Update .env with new password
vim .env

# 2. Update PostgreSQL password
psql -U postgres -c "ALTER USER your_user WITH PASSWORD 'new_password';"

# 3. Restart services
docker-compose restart

# 4. Verify connection
python -c "from rag_web import test_db_connection; print(test_db_connection())"
```

## SQL Injection Prevention

### Safe Database Queries

**Always use parameterized queries** with `psycopg2.sql.Identifier()`:

**Correct** ✓
```python
from psycopg2 import sql

# Safe: Table name properly escaped
cur.execute(
    sql.SQL("SELECT COUNT(*) FROM {}").format(
        sql.Identifier(table_name)
    )
)

# Safe: Values parameterized
cur.execute(
    "SELECT * FROM users WHERE id = %s",
    (user_id,)
)
```

**Wrong** ✗
```python
# VULNERABLE to SQL injection!
cur.execute(f"SELECT COUNT(*) FROM {table_name}")
cur.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

### Why This Matters

An attacker could exploit f-string queries:
```python
# If table_name = "users; DROP TABLE users--"
cur.execute(f"SELECT * FROM {table_name}")  # Executes: DROP TABLE users
```

With parameterized queries, the attack is neutralized:
```python
# Safely treats entire input as identifier
cur.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name)))
```

## Code Injection Prevention

### Never Use eval()

**Correct** ✓
```python
import ast

# Safe: Only parses Python literals
data = ast.literal_eval("[1, 2, 3]")
```

**Wrong** ✗
```python
# DANGEROUS: Can execute arbitrary code!
data = eval("[1, 2, 3]")  # What if input is "__import__('os').system('rm -rf /')"?
```

### Safe Parsing

For JSON:
```python
import json
data = json.loads(json_string)
```

For Python literals:
```python
import ast
data = ast.literal_eval(literal_string)
```

For YAML (with safe loader):
```python
import yaml
data = yaml.safe_load(yaml_string)
```

## Web UI Security

### Authentication Setup

1. **Install streamlit-authenticator**:
   ```bash
   pip install streamlit-authenticator
   ```

2. **Generate password hash**:
   ```python
   import streamlit_authenticator as stauth

   hashed_passwords = stauth.Hasher(['password123']).generate()
   print(hashed_passwords)
   ```

3. **Create config file** (`config/auth.yaml`):
   ```yaml
   credentials:
     usernames:
       admin:
         name: Admin User
         password: $2b$12$... # hashed password
       readonly:
         name: Read Only User
         password: $2b$12$...
   cookie:
     name: rag_auth_cookie
     key: your_secret_key_here  # Generate with secrets.token_urlsafe(32)
     expiry_days: 30
   ```

4. **Add to rag_web.py**:
   ```python
   import streamlit_authenticator as stauth
   import yaml

   # Load config
   with open('config/auth.yaml') as file:
       config = yaml.safe_load(file)

   # Initialize authenticator
   authenticator = stauth.Authenticate(
       config['credentials'],
       config['cookie']['name'],
       config['cookie']['key'],
       config['cookie']['expiry_days']
   )

   # Login widget
   name, authentication_status, username = authenticator.login('Login', 'main')

   if authentication_status:
       authenticator.logout('Logout', 'sidebar')
       st.sidebar.write(f'Welcome *{name}*')
       # Rest of your app...
   elif authentication_status == False:
       st.error('Username/password is incorrect')
   elif authentication_status == None:
       st.warning('Please enter your username and password')
   ```

### HTTPS/TLS

For production, always use HTTPS:

```bash
# Using Streamlit with SSL
streamlit run rag_web.py \
  --server.sslCertFile=/path/to/cert.pem \
  --server.sslKeyFile=/path/to/key.pem
```

Or use a reverse proxy (nginx, Caddy):
```nginx
server {
    listen 443 ssl;
    server_name rag.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Database Security

### PostgreSQL Hardening

1. **Use SSL/TLS connections**:
   ```python
   conn = psycopg2.connect(
       host=os.environ['PGHOST'],
       port=os.environ['PGPORT'],
       user=os.environ['PGUSER'],
       password=os.environ['PGPASSWORD'],
       dbname=os.environ['DB_NAME'],
       sslmode='require'  # Enforce SSL
   )
   ```

2. **Configure pg_hba.conf**:
   ```conf
   # Allow SSL connections only
   hostssl all all 0.0.0.0/0 scram-sha-256

   # Reject non-SSL
   host    all all 0.0.0.0/0 reject
   ```

3. **Enable audit logging** (`postgresql.conf`):
   ```conf
   log_statement = 'all'
   log_connections = on
   log_disconnections = on
   log_duration = on
   ```

4. **Restrict network access**:
   ```bash
   # Docker Compose - internal network only
   networks:
     rag_network:
       internal: true  # No external access
   ```

### Principle of Least Privilege

Create separate users with minimal permissions:

```sql
-- Read-only user for queries
CREATE USER rag_reader WITH PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE vector_db TO rag_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_reader;

-- Write user for indexing
CREATE USER rag_writer WITH PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE vector_db TO rag_writer;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO rag_writer;

-- Admin user (minimal use)
CREATE USER rag_admin WITH PASSWORD 'strong_password';
GRANT ALL PRIVILEGES ON DATABASE vector_db TO rag_admin;
```

Use appropriate user for each operation:
```bash
# Read-only queries
PGUSER=rag_reader python rag_interactive.py

# Indexing operations
PGUSER=rag_writer python rag_low_level_m1_16gb_verbose.py

# Database maintenance
PGUSER=rag_admin psql
```

## Dependency Security

### Regular Updates

```bash
# Check for vulnerabilities
pip-audit

# Check for outdated packages
pip list --outdated

# Update packages
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Verify no new vulnerabilities
pip-audit
```

### Security Scanning

Run before each commit:

```bash
# Static analysis with Bandit
bandit -r . -ll -f json -o security_scan.json

# Dependency audit
pip-audit --format json --output pip-audit.json

# Check for known vulnerabilities
safety check --json > safety-report.json
```

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.5'
    hooks:
      - id: bandit
        args: ['-ll', '-i']

  - repo: https://github.com/pypa/pip-audit
    rev: 'v2.6.1'
    hooks:
      - id: pip-audit
```

Install hooks:
```bash
pip install pre-commit
pre-commit install
```

## Network Security

### Firewall Rules

Restrict access to services:

```bash
# PostgreSQL - localhost only
PGPORT=127.0.0.1:5432:5432

# Streamlit - localhost only (use reverse proxy for external access)
streamlit run rag_web.py --server.address localhost
```

### Docker Network Isolation

```yaml
# docker-compose.yml
services:
  db:
    networks:
      - backend  # Internal network only

  app:
    networks:
      - backend
      - frontend

  web:
    networks:
      - frontend
    ports:
      - "443:443"  # Only expose web UI

networks:
  backend:
    internal: true  # No external access
  frontend:
    driver: bridge
```

## Audit Logging

### Query Logging

Enable query logging:
```python
# In rag_low_level_m1_16gb_verbose.py
LOG_QUERIES = os.environ.get("LOG_QUERIES", "0") == "1"

if LOG_QUERIES:
    log_query(query, user, timestamp, results)
```

### Access Logging

Log all authentication attempts:
```python
# In rag_web.py
def log_access(username, success, ip_address):
    timestamp = datetime.now().isoformat()
    with open("logs/access.log", "a") as f:
        f.write(f"{timestamp},{username},{success},{ip_address}\n")
```

### Security Event Logging

Log security-relevant events:
- Authentication failures
- Permission denials
- SQL injection attempts
- Unusual query patterns
- Configuration changes

## Incident Response

### Detection

Monitor for:
- Multiple failed login attempts
- Unusual database queries
- High CPU/memory usage
- Large data exports
- Suspicious network traffic

### Response Plan

1. **Isolate**: Disconnect affected systems
2. **Investigate**: Review logs, identify attack vector
3. **Contain**: Patch vulnerability, rotate credentials
4. **Recover**: Restore from backup if needed
5. **Document**: Write incident report
6. **Improve**: Update security measures

### Emergency Contacts

Maintain list of:
- Security team contacts
- Database administrators
- Cloud provider support
- Incident response team

## Backup Security

### Encrypted Backups

```bash
# Backup with encryption
pg_dump -U postgres vector_db | \
  gpg --encrypt --recipient your-key@example.com > \
  backup.sql.gpg

# Restore
gpg --decrypt backup.sql.gpg | \
  psql -U postgres vector_db
```

### Secure Storage

- Store backups off-site (cloud storage)
- Encrypt at rest (GPG, age, or cloud KMS)
- Encrypt in transit (TLS/SSL)
- Test restoration regularly
- Implement backup retention policy (7-30 days)
- Restrict access (separate credentials)

## Compliance

### Data Privacy

- Minimize PII in logs
- Implement data retention policies
- Honor deletion requests (GDPR, CCPA)
- Encrypt sensitive data
- Document data flows

### Audit Requirements

- Maintain access logs (1+ year)
- Regular security assessments
- Vulnerability management
- Incident response documentation
- Security training records

## Security Checklist for Production

- [ ] All credentials in environment variables
- [ ] Strong passwords (16+ characters)
- [ ] Web UI authentication enabled
- [ ] HTTPS/TLS configured
- [ ] Database SSL connections
- [ ] Firewall rules configured
- [ ] Network segmentation
- [ ] Audit logging enabled
- [ ] Backup encryption
- [ ] Security scanning automated
- [ ] Dependency updates scheduled
- [ ] Incident response plan documented
- [ ] Security contacts listed
- [ ] Pre-commit hooks installed
- [ ] Monitoring alerts configured

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)
- [Streamlit Security](https://docs.streamlit.io/knowledge-base/deploy/authentication)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

## Contact

For security vulnerabilities, contact:
- Email: security@example.com
- Do NOT create public GitHub issues for security bugs
- Use responsible disclosure

---

**Remember**: Security is not a one-time task, it's an ongoing process. Review and update security measures regularly.
