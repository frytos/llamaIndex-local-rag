# Phase 2: API Security Layer - Research

**Researched:** 2026-01-15
**Domain:** HTTPS/SSL automation, security headers, data encryption for Python web applications
**Confidence:** HIGH

<research_summary>
## Summary

Researched the HTTPS/SSL ecosystem for securing a Python Streamlit web application with automated certificate management, HTTPS enforcement, and data encryption. The standard approach uses reverse proxies (Caddy or Nginx+Certbot) for SSL termination, with Caddy being the modern zero-configuration choice.

Key finding: Don't hand-roll certificate management or HTTPS configuration. Caddy provides automatic HTTPS with Let's Encrypt out of the box, handling certificate acquisition, renewal, and HTTP-to-HTTPS redirects without configuration. For traditional setups, Certbot automates Let's Encrypt certificates for Nginx/Apache.

**Primary recommendation:** Use Caddy as reverse proxy for Streamlit app. Zero-config automatic HTTPS, certificate renewal, and HTTP redirects. Add security headers middleware in Python for defense-in-depth. PostgreSQL SSL connections for data in transit, filesystem encryption for data at rest.
</research_summary>

<standard_stack>
## Standard Stack

The established libraries/tools for HTTPS and security:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Caddy | 2.x | Reverse proxy with automatic HTTPS | Zero-config Let's Encrypt, modern TLS defaults |
| Certbot | 5.2.2+ | Let's Encrypt automation (Nginx/Apache) | EFF-maintained, official ACME client |
| secure.py | 0.3.0+ | Python security headers middleware | Comprehensive headers with framework support |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PostgreSQL SSL | 18+ | Encrypted database connections | Always for production PostgreSQL |
| pgcrypto | Built-in | Column-level encryption | For sensitive fields in database |
| LUKS/dm-crypt | System | Disk encryption | For data at rest on Linux |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Caddy | Nginx + Certbot | Nginx more flexible but requires manual HTTPS config |
| Caddy | Traefik | Traefik better for Docker/K8s orchestration |
| Let's Encrypt | Self-signed certs | Self-signed = browser warnings, no trust |

**Installation:**

**Option 1: Caddy (Recommended for simplicity)**
```bash
# Install Caddy (Ubuntu/Debian)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# Caddyfile for Streamlit app
echo "your-domain.com {
    reverse_proxy localhost:8501
}" | sudo tee /etc/caddy/Caddyfile

sudo systemctl restart caddy
```

**Option 2: Nginx + Certbot (Traditional approach)**
```bash
# Install Nginx and Certbot
sudo apt install nginx python3-certbot-nginx

# Get certificate and configure Nginx
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up via systemd timer
```

**Python security headers:**
```bash
pip install secure
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```
├── app.py                    # Streamlit app
├── auth_middleware.py        # Authentication from Phase 1
├── security_middleware.py    # Security headers middleware
├── config/
│   ├── Caddyfile            # Caddy reverse proxy config
│   └── .env                 # Environment variables (DB creds, etc.)
└── deploy/
    └── caddy-setup.sh       # Deployment script
```

### Pattern 1: Reverse Proxy SSL Termination (Recommended)
**What:** Caddy/Nginx handles HTTPS, forwards HTTP to Streamlit app
**When to use:** Production deployments (always recommended)
**Example:**
```caddyfile
# Caddyfile - Automatic HTTPS with Let's Encrypt
your-domain.com {
    # Caddy automatically:
    # - Gets Let's Encrypt certificate
    # - Renews certificates before expiry
    # - Redirects HTTP -> HTTPS
    # - Configures modern TLS settings

    reverse_proxy localhost:8501 {
        # Forward to Streamlit app
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }
}
```

**Why this pattern:**
- **Zero config HTTPS**: Caddy automatically obtains and renews Let's Encrypt certificates
- **Separation of concerns**: Streamlit app doesn't handle TLS
- **Standard architecture**: Industry best practice for web apps
- **Easy updates**: Update Streamlit without touching SSL config

### Pattern 2: Security Headers Middleware
**What:** Add security headers via Python middleware
**When to use:** Defense-in-depth with reverse proxy
**Example:**
```python
# security_middleware.py
from secure import Secure

secure_headers = Secure(
    hsts=Secure.StrictTransportSecurity().include_subdomains().preload().max_age(63072000),
    csp=Secure.ContentSecurityPolicy().default_src("'self'"),
    xfo=Secure.XFrameOptions().deny(),
    xxp=Secure.XXSSProtection().enabled_block(),
    referrer=Secure.ReferrerPolicy().strict_origin_when_cross_origin(),
)

# For Streamlit, inject headers via st.markdown with unsafe_allow_html
# Or use Caddy/Nginx to add headers (cleaner approach)
```

**Better: Add headers in Caddy**
```caddyfile
your-domain.com {
    reverse_proxy localhost:8501

    header {
        Strict-Transport-Security "max-age=63072000; includeSubDomains; preload"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
        Permissions-Policy "geolocation=(), camera=(), microphone=()"
    }
}
```

### Pattern 3: PostgreSQL SSL Connections
**What:** Enforce SSL for database connections
**When to use:** Always in production
**Example:**
```python
# Connection string with SSL
import psycopg2

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    dbname=os.getenv("DB_NAME"),
    sslmode='require'  # Options: disable, allow, prefer, require, verify-ca, verify-full
)
```

**pg_hba.conf on PostgreSQL server:**
```
# Require SSL for remote connections
hostssl    all    all    0.0.0.0/0    scram-sha-256
```

### Anti-Patterns to Avoid
- **Terminating SSL in Streamlit app**: Streamlit `server.sslCertFile` is for testing only, not production
- **Self-signed certificates**: Browsers show warnings, users get trained to bypass security
- **Manual certificate renewal**: Use Caddy or Certbot systemd timers for automatic renewal
- **Mixed content**: Serving HTTP resources from HTTPS pages breaks security
- **No HSTS header**: Allows downgrade attacks even with HTTPS
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Certificate management | Custom ACME client | Caddy or Certbot | ACME protocol is complex, renewal timing, challenge types |
| HTTPS configuration | Manual TLS settings | Caddy defaults or Mozilla SSL Config Generator | Cipher suites, protocols, vulnerabilities evolve |
| Certificate renewal | Cron job with custom script | Caddy auto-renewal or Certbot systemd timer | Zero-downtime renewal, error handling, retries |
| HTTP-to-HTTPS redirects | Custom redirect logic | Caddy automatic or Nginx config | Edge cases (HSTS, HTTPS-only cookies, websockets) |
| Security headers | Manual header setting | Caddy header directive or secure.py | Complete set, correct syntax, maintenance |

**Key insight:** SSL/TLS is a security-critical domain with decades of evolution. Caddy represents the modern approach: security by default, automatic certificate lifecycle. Certbot is the battle-tested traditional approach. Both are maintained by experts who track vulnerabilities and best practices. Custom implementations inevitably miss edge cases or fall behind on security updates.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Certificate Expiration
**What goes wrong:** Let's Encrypt certificates expire after 90 days, site becomes inaccessible
**Why it happens:** Manual renewal forgotten, auto-renewal failed silently
**How to avoid:**
- **Caddy**: Automatic renewal runs in background, no action needed
- **Certbot**: Verify systemd timer is enabled: `systemctl status certbot.timer`
- Set up monitoring/alerts for certificate expiry (30-day warning)
**Warning signs:** Certificate expiry date approaching (check with `openssl s_client -connect domain:443 -servername domain`)

### Pitfall 2: HSTS with Self-Signed Certs
**What goes wrong:** Browser refuses to connect, "NET::ERR_CERT_AUTHORITY_INVALID" can't be bypassed
**Why it happens:** HSTS header sent with self-signed cert, browser enforces it strictly
**How to avoid:**
- **Never use HSTS with self-signed certificates**
- Only enable HSTS after Let's Encrypt certificate is working
- For local dev, use `localhost` without HSTS
**Warning signs:** Unable to access site even after removing HSTS header (browser cached it)

### Pitfall 3: Port 80 Not Open for ACME Challenge
**What goes wrong:** Let's Encrypt certificate acquisition fails, HTTPS doesn't work
**Why it happens:** HTTP-01 challenge requires port 80 open, firewall blocks it
**How to avoid:**
- **Ensure port 80 and 443 open** on firewall/cloud security groups
- Use DNS-01 challenge if port 80 unavailable (requires DNS provider API access)
- Caddy will try TLS-ALPN challenge (port 443) as fallback
**Warning signs:** ACME errors in logs: "connection refused", "timeout"

### Pitfall 4: Database SSL Misconfiguration
**What goes wrong:** Connection fails with "SSL required" or data transmitted unencrypted
**Why it happens:** Client `sslmode` doesn't match server `pg_hba.conf` requirements
**How to avoid:**
- Start with `sslmode=prefer` (tries SSL, falls back to non-SSL)
- Move to `sslmode=require` in production
- Use `sslmode=verify-full` for maximum security (validates cert chain)
**Warning signs:** Connection errors mentioning SSL, or successful connections without SSL in logs

### Pitfall 5: Reverse Proxy Forwarding Headers Missing
**What goes wrong:** Streamlit app sees all connections from 127.0.0.1, logs wrong IPs
**Why it happens:** Reverse proxy doesn't set X-Forwarded-For, X-Real-IP headers
**How to avoid:**
- Configure Caddy to set `X-Real-IP` and `X-Forwarded-Proto` (see Pattern 1)
- In Streamlit, trust proxy headers for correct remote IP
**Warning signs:** Authentication logs show localhost IP for all users
</common_pitfalls>

<code_examples>
## Code Examples

Verified patterns from official sources:

### Caddy Automatic HTTPS Setup
```caddyfile
# Source: https://caddyserver.com/docs/automatic-https
# /etc/caddy/Caddyfile

# Simple reverse proxy with automatic HTTPS
your-domain.com {
    reverse_proxy localhost:8501
}

# With security headers
your-domain.com {
    reverse_proxy localhost:8501 {
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }

    header {
        Strict-Transport-Security "max-age=63072000; includeSubDomains; preload"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
        Content-Security-Policy "default-src 'self'"
        Permissions-Policy "geolocation=(), camera=(), microphone=()"
        -Server  # Remove server header
    }

    # Optional: Enable compression
    encode gzip
}
```

### Certbot with Nginx
```bash
# Source: https://eff-certbot.readthedocs.io/
# Get certificate and auto-configure Nginx
sudo certbot --nginx -d your-domain.com

# Manual renewal (not needed, systemd timer handles it)
sudo certbot renew

# Test renewal process
sudo certbot renew --dry-run

# Set up automatic renewal with hooks
# /etc/letsencrypt/renewal-hooks/deploy/restart-app.sh
#!/bin/bash
systemctl reload nginx
systemctl restart streamlit-app
```

### PostgreSQL SSL Connection
```python
# Source: https://www.postgresql.org/docs/current/libpq-ssl.html
import psycopg2
import os

# Connection with SSL enforcement
conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT", 5432),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    dbname=os.getenv("DB_NAME"),
    sslmode='require',  # Options: disable, allow, prefer, require, verify-ca, verify-full
    connect_timeout=10
)
conn.autocommit = True

# Verify SSL connection
with conn.cursor() as cur:
    cur.execute("SELECT version(), pg_is_in_recovery(), ssl_is_used();")
    version, is_replica, ssl_used = cur.fetchone()
    print(f"PostgreSQL {version}")
    print(f"SSL enabled: {ssl_used}")
```

### Environment Variables for SSL Config
```bash
# .env file
# Database with SSL
PGHOST=your-postgres-host.com
PGPORT=5432
PGUSER=your_user
PGPASSWORD=your_password
DB_NAME=vector_db
PGSSLMODE=require  # or verify-full with CA cert

# Streamlit app
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost  # Only bind to localhost, Caddy handles external

# Domain for certificates (used in deployment script)
DOMAIN=your-domain.com
```
</code_examples>

<sota_updates>
## State of the Art (2026)

What's changed recently:

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual cert renewal | Caddy automatic HTTPS | 2016+ | Zero-config SSL, self-renewing certs |
| Certbot manual setup | Certbot systemd timer | 2018+ | Automatic renewal without cron |
| TLS 1.0/1.1 | TLS 1.2+ minimum | 2020+ | PCI DSS requirement, modern security |
| HTTP/1.1 only | HTTP/2, HTTP/3 support | 2022+ | Performance, multiplexing |
| Let's Encrypt only | Let's Encrypt + ZeroSSL fallback | 2020+ | Redundancy (Caddy default) |

**New tools/patterns to consider:**
- **Encrypted ClientHello (ECH)**: Caddy supports ECH to hide domain names from network observers (privacy enhancement)
- **Wildcard certificates**: Let's Encrypt via DNS-01 challenge for `*.domain.com` (useful for subdomains)
- **On-demand TLS**: Caddy can obtain certificates dynamically during first connection (multi-tenant apps)
- **ACME External Account Binding**: Required by some CAs for business accounts

**Deprecated/outdated:**
- **TLS 1.0/1.1**: Disabled by modern browsers, insecure
- **SHA-1 certificates**: No longer trusted
- **Manual certificate management**: Automated renewal is now standard
- **HTTP without redirect**: Should always redirect to HTTPS in production
</sota_updates>

<open_questions>
## Open Questions

Things that couldn't be fully resolved:

1. **Deployment platform SSL handling**
   - What we know: Vercel/Railway provide automatic HTTPS for deployed apps
   - What's unclear: Whether custom domain HTTPS conflicts with Caddy setup
   - Recommendation: If using Vercel/Railway, leverage their HTTPS. Use Caddy for self-hosted VPS deployment

2. **pgvector-specific encryption**
   - What we know: pgvector stores embeddings as PostgreSQL arrays/columns
   - What's unclear: Performance impact of pgcrypto column encryption on vector similarity search
   - Recommendation: Use PostgreSQL SSL for transit, filesystem encryption for rest. Column encryption may break pgvector indexing

3. **Streamlit session security with HTTPS**
   - What we know: Streamlit uses session cookies from streamlit-authenticator (Phase 1)
   - What's unclear: Whether session cookies need explicit secure/httponly flags
   - Recommendation: Verify session cookies have `Secure` flag when behind HTTPS proxy. May need middleware adjustment
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- Context7: /certbot/certbot - Certificate automation, renewal, Nginx integration
- https://letsencrypt.org/getting-started/ - Official Let's Encrypt guide
- https://eff-certbot.readthedocs.io/ - Certbot documentation v5.2.2
- https://caddyserver.com/docs/automatic-https - Caddy automatic HTTPS (verified 2026-01-15)
- https://www.postgresql.org/docs/current/encryption-options.html - PostgreSQL 18 encryption guide
- https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Headers_Cheat_Sheet.html - OWASP security headers

### Secondary (MEDIUM confidence)
- [Caddy automatic HTTPS overview](https://caddyserver.com/) - Verified with docs, modern web server standards
- [Streamlit HTTPS deployment](https://docs.streamlit.io/develop/concepts/configuration/https-support) - Official Streamlit docs, recommends reverse proxy
- [PostgreSQL security best practices](https://www.tigerdata.com/learn/postgres-security-best-practices) - Cross-verified with PostgreSQL docs
- [Python secure headers library](https://pypi.org/project/secure/0.1.5/) - Active maintenance through 2026

### Tertiary (LOW confidence - needs validation)
- None - all findings verified with official sources
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: Let's Encrypt + ACME protocol
- Ecosystem: Caddy, Certbot, Nginx, PostgreSQL SSL, security headers
- Patterns: Reverse proxy SSL termination, automatic renewal, database encryption
- Pitfalls: Certificate expiry, HSTS misconfiguration, port 80 access, SSL modes

**Confidence breakdown:**
- Standard stack: HIGH - Caddy and Certbot are industry standards, official documentation
- Architecture: HIGH - Reverse proxy pattern is universal best practice
- Pitfalls: HIGH - Well-documented issues, verified in multiple sources
- Code examples: HIGH - From official Caddy, Certbot, PostgreSQL documentation

**Research date:** 2026-01-15
**Valid until:** 2026-02-15 (30 days - SSL/TLS ecosystem is stable)

**Research methodology:**
- Context7 for Certbot library documentation
- Official documentation (Let's Encrypt, Caddy, PostgreSQL, OWASP)
- WebSearch for ecosystem trends and recent updates (2026)
- Cross-verification of all claims with authoritative sources
</metadata>

---

*Phase: 02-api-security-layer*
*Research completed: 2026-01-15*
*Ready for planning: yes*
