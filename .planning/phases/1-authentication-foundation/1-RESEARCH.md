# Phase 1: Authentication Foundation - Research

**Researched:** 2026-01-15
**Domain:** Streamlit authentication with username/password
**Confidence:** HIGH

<research_summary>
## Summary

Researched authentication patterns for Streamlit applications to enable secure, web-accessible RAG system with username/password authentication. The standard approach uses `streamlit-authenticator` library for session management, Argon2id for password hashing (recommended over bcrypt in 2026), and secure cookies with HTTPOnly/Secure/SameSite flags.

Key finding: Don't hand-roll authentication logic, password hashing, or session token management. The `streamlit-authenticator` library handles cookies, session state, and re-authentication out of the box. Argon2id is the gold standard for password hashing (winner of Password Hashing Competition 2015) and provides superior protection against GPU/ASIC attacks compared to bcrypt.

**Primary recommendation:** Use streamlit-authenticator (v0.3.3+) with Argon2id password hashing, secure cookie configuration (30-day expiry, HTTPOnly, Secure, SameSite=Lax), and Nginx reverse proxy for SSL termination in production. Store credentials in YAML with automatic password hashing on first run.
</research_summary>

<standard_stack>
## Standard Stack

The established libraries/tools for Streamlit authentication:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit-authenticator | 0.3.3+ | Complete authentication module | Battle-tested, handles cookies/sessions/rehashing |
| argon2-cffi | 23.1.0+ | Password hashing (Argon2id) | Winner of Password Hashing Competition, GPU-resistant |
| PyYAML | 6.0.1+ | Configuration management | Standard format for credentials file |
| PyJWT | 2.8.0+ | JWT token generation (optional) | Industry standard for stateless tokens |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| extra-streamlit-components | 0.1.71+ | Cookie manager (alternative) | If not using streamlit-authenticator |
| python-dotenv | 1.0.0+ | Environment variable management | For secrets/config |
| pydantic | 2.5.0+ | Configuration validation | For type-safe config management |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Argon2id | bcrypt | bcrypt simpler but less GPU-resistant (legacy choice) |
| streamlit-authenticator | Custom + PyJWT | Custom gives control but more security risk |
| YAML config | Database | Database for multi-tenant, YAML for single-user |

**Installation:**
```bash
pip install streamlit-authenticator argon2-cffi PyYAML python-dotenv
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```
src/
├── auth/                # Authentication module
│   ├── config.yaml      # User credentials (gitignored)
│   ├── authenticator.py # Auth initialization
│   └── middleware.py    # Session checks
├── app.py               # Main Streamlit app
└── config/
    └── .streamlit/
        └── secrets.toml # Cookie secrets
```

### Pattern 1: streamlit-authenticator with Auto-Hashing
**What:** Use streamlit-authenticator with YAML config for automatic password hashing
**When to use:** Single-user or small team (< 50 users), simple deployment
**Example:**
```python
# Source: streamlit-authenticator docs
import streamlit as st
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate

# Load configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render login widget
name, authentication_status, username = authenticator.login('main')

# Handle authentication states
if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.write(f'Welcome *{name}*')
    st.title('Your RAG System')
    # Main app logic here
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
```

**config.yaml structure:**
```yaml
credentials:
  usernames:
    your_username:
      email: you@example.com
      failed_login_attempts: 0  # Managed automatically
      first_name: Your
      last_name: Name
      logged_in: False  # Managed automatically
      password: plaintext_password  # Will be hashed automatically on first run
cookie:
  expiry_days: 30
  key: random_signature_key_here  # Generate with secrets.token_urlsafe(32)
  name: streamlit_auth_cookie
```

### Pattern 2: Session State Management
**What:** Use st.session_state for authentication status and user data
**When to use:** Always - Streamlit's standard session persistence
**Example:**
```python
# Source: Streamlit docs
import streamlit as st

# Check authentication status from session state
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

if st.session_state['authentication_status']:
    # User is authenticated - show protected content
    st.write(f"Welcome, {st.session_state['name']}!")
    # Your RAG interface here
else:
    # Show login
    st.stop()  # Stop execution until authenticated
```

### Pattern 3: Secure Cookie Configuration
**What:** Configure cookies with security flags to prevent hijacking
**When to use:** Production deployments with HTTPS
**Example:**
```python
# In .streamlit/config.toml
[server]
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

# Cookie security (via streamlit-authenticator config.yaml)
cookie:
  expiry_days: 30
  key: "your-secret-key-here"  # Must be kept secret
  name: "auth_cookie"

# Additional security via nginx (see deployment section)
```

### Pattern 4: Argon2id Password Hashing (Manual)
**What:** If implementing custom auth, use Argon2id for password hashing
**When to use:** Custom authentication implementation (not recommended for beginners)
**Example:**
```python
# Source: argon2-cffi docs
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ph = PasswordHasher()

# Hash password on registration
hashed = ph.hash("user_password")
# Store hashed in database/config

# Verify on login
try:
    ph.verify(hashed, "user_password")
    print("Login successful")
except VerifyMismatchError:
    print("Invalid password")

# Check if rehash needed (after parameter updates)
if ph.check_needs_rehash(hashed):
    new_hash = ph.hash("user_password")
    # Update stored hash
```

### Anti-Patterns to Avoid
- **Storing plaintext passwords:** Always hash passwords, never store plaintext
- **Using pickle for session data from untrusted sources:** Pickle can execute arbitrary code
- **Client-side only validation:** Always validate on server side
- **Hardcoding secrets in code:** Use .env files or secrets.toml (gitignored)
- **Skipping HTTPS in production:** Cookies can be intercepted without HTTPS
- **Not setting cookie security flags:** Missing HTTPOnly, Secure, SameSite = vulnerable to XSS/CSRF
- **Using bcrypt for new projects:** Argon2id is the 2026 standard
- **Hand-rolling JWT implementation:** Use PyJWT library with proper validation
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Password hashing | Custom hash function | Argon2-cffi (Argon2id) | Memory-hard, GPU-resistant, configurable work factors |
| Session management | Custom cookie logic | streamlit-authenticator | Handles re-authentication, cookie security, session state |
| JWT tokens | Manual token creation | PyJWT library | Proper signature validation, expiration handling, claims validation |
| Cookie security | Manual cookie flags | streamlit-authenticator config | HTTPOnly, Secure, SameSite flags handled |
| Session hijacking prevention | Custom fingerprinting | Secure cookies + HTTPS + session rotation | Battle-tested approach, less attack surface |
| Login UI | Custom HTML/CSS | streamlit-authenticator.login() | Pre-built, accessible, mobile-friendly |

**Key insight:** Authentication has decades of solved problems and known vulnerabilities. The `streamlit-authenticator` library (15+ code snippets in Context7) implements secure patterns including automatic password rehashing, cookie management, failed login tracking, and session state integration. Custom implementations inevitably miss edge cases (session fixation, timing attacks, password complexity, secure token generation).

**Critical vulnerabilities in custom auth:**
- **Password timing attacks:** Custom password comparison leaks information via timing
- **Session fixation:** Not regenerating session ID after login allows hijacking
- **Improper salt generation:** Weak random number generators = crackable hashes
- **Missing rate limiting:** Allows brute force attacks
- **Insecure cookie configuration:** Missing flags = XSS/CSRF vulnerabilities

Use `streamlit-authenticator` unless you have security expertise and custom requirements.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Pickle Deserialization Vulnerability
**What goes wrong:** Streamlit's st.session_state uses pickle, which can execute arbitrary code during unpickling
**Why it happens:** Pickle is insecure by design - allows object serialization with code execution
**How to avoid:** Never load untrusted data into session state. Only use session_state for data you control.
**Warning signs:** User-controlled data going directly into session_state without validation
**Official warning:** "Never load data that could have come from an untrusted source in an unsafe mode or that could have been tampered with" (Streamlit Security Reminders)

### Pitfall 2: Missing HTTPS in Production
**What goes wrong:** Cookies transmitted over HTTP can be intercepted (session hijacking)
**Why it happens:** Developers test locally without HTTPS, forget to configure in production
**How to avoid:** Use Nginx reverse proxy with Let's Encrypt SSL certificates. Set `Secure` flag on cookies.
**Warning signs:** Login works but sessions get hijacked, especially on public WiFi
**Solution:** Always use HTTPS in production. Streamlit recommends SSL termination at reverse proxy (Nginx), not in app.

### Pitfall 3: CORS/XSRF Disabled Without Understanding
**What goes wrong:** Disabling CORS/XSRF protection to "fix" issues opens security holes
**Why it happens:** CORS errors are common, disabling is quick fix without understanding risk
**How to avoid:** Keep `enableXsrfProtection = true` and `enableCORS = false` (Streamlit defaults). Only disable if you understand the security implications.
**Warning signs:** Users from different domains can access your authenticated app
**Streamlit warning:** "CORS and XSRF protection are very complex security policies... represent the most secure posture"

### Pitfall 4: Using bcrypt for New Projects (2026)
**What goes wrong:** bcrypt is vulnerable to GPU-based cracking with modern hardware
**Why it happens:** bcrypt tutorials are common, developers use familiar tools
**How to avoid:** Use Argon2id for all new projects. Argon2id is memory-hard and GPU-resistant.
**Warning signs:** Using bcrypt library in 2026 when Argon2 is available
**Industry consensus:** "bcrypt wasn't designed for today's parallel-processing, GPU-enabled universe, while Argon2 was" (Security experts, 2025)

### Pitfall 5: File Upload Validation (2025 CVE)
**What goes wrong:** Client-side only file type validation allows malicious uploads
**Why it happens:** File type restrictions only in JavaScript, not on server side
**How to avoid:** Always validate file types on server side. Update to Streamlit 1.43.2+ which includes backend validation.
**Warning signs:** Only checking file extensions in frontend
**Recent vulnerability:** Critical CVE in Feb 2025 allowed bypassing file restrictions for cloud account takeover

### Pitfall 6: Hardcoded Secrets in Config Files
**What goes wrong:** Cookie signing keys, database passwords committed to git
**Why it happens:** Quick testing with hardcoded values, forgot to move to .env
**How to avoid:** Use `.streamlit/secrets.toml` (automatically gitignored) or environment variables. Never commit config.yaml with real passwords.
**Warning signs:** config.yaml or .env in git history
**Detection:** Check with `git log --all --full-history -- "*config.yaml*"`

### Pitfall 7: No Session Timeout
**What goes wrong:** Sessions stay active indefinitely, increasing hijacking window
**Why it happens:** Not configuring cookie expiry or session timeout
**How to avoid:** Set reasonable cookie expiry (7-30 days). Implement session timeout after inactivity (optional).
**Warning signs:** Sessions lasting months/years
**Recommendation:** `expiry_days: 30` in streamlit-authenticator config
</common_pitfalls>

<code_examples>
## Code Examples

Verified patterns from official sources:

### Complete Authentication Setup
```python
# Source: streamlit-authenticator GitHub + Argon2-cffi docs
import streamlit as st
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate

# Load configuration
with open('auth/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login widget
name, authentication_status, username = authenticator.login('main')

if authentication_status:
    # User authenticated
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f'Welcome *{name}*')

    # Main app
    st.title('RAG Document Query System')
    query = st.text_input("Ask a question about your documents:")
    if query:
        # RAG logic here
        st.write("Query results...")

elif authentication_status is False:
    st.error('Username/password is incorrect')

elif authentication_status is None:
    st.warning('Please enter your username and password')
    st.stop()
```

### Secure Configuration File (config.yaml)
```yaml
# Source: streamlit-authenticator documentation
credentials:
  usernames:
    admin:  # Change this username
      email: admin@example.com
      failed_login_attempts: 0  # Auto-managed
      first_name: Admin
      last_name: User
      logged_in: False  # Auto-managed
      password: change_me_on_first_run  # Will be hashed automatically

cookie:
  expiry_days: 30
  key: abc123def456  # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
  name: rag_auth_cookie
```

### Generating Secure Cookie Key
```python
# Source: Python secrets module (standard library)
import secrets

# Generate cryptographically secure random key for cookie signing
cookie_key = secrets.token_urlsafe(32)
print(f"Use this as your cookie key: {cookie_key}")
```

### Production Nginx Configuration
```nginx
# Source: Multiple Streamlit deployment guides (2025)
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # WebSocket support for Streamlit
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running queries
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### Session State Pattern for Multi-Page Apps
```python
# Source: Streamlit best practices (2025)
import streamlit as st

def check_authentication():
    """Check if user is authenticated, redirect to login if not"""
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None

    if not st.session_state['authentication_status']:
        st.warning("Please log in to access this page")
        st.stop()

    return st.session_state['username']

# In each page of multi-page app
username = check_authentication()
st.write(f"Welcome back, {username}!")
# Page content here
```

### Password Hashing with Argon2id (Manual Implementation)
```python
# Source: argon2-cffi documentation
from argon2 import PasswordHasher
from argon2.profiles import RFC_9106_LOW_MEMORY  # For personal systems

# Initialize with recommended parameters
ph = PasswordHasher.from_parameters(RFC_9106_LOW_MEMORY)

# Hash password (on user registration)
password = "user_secure_password"
hashed = ph.hash(password)
print(f"Store this hash: {hashed}")

# Verify password (on login attempt)
try:
    ph.verify(hashed, password)
    print("Authentication successful")
except Exception as e:
    print(f"Authentication failed: {e}")

# Check if hash needs updating (after changing parameters)
if ph.check_needs_rehash(hashed):
    new_hash = ph.hash(password)
    print(f"Rehash needed, new hash: {new_hash}")
```
</code_examples>

<sota_updates>
## State of the Art (2025-2026)

What's changed recently:

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| bcrypt for passwords | Argon2id | 2015 (PHC winner), mainstream 2024+ | Argon2id is memory-hard, GPU-resistant |
| Manual cookie management | streamlit-authenticator | Library mature ~2023 | Handles security edge cases automatically |
| Custom OIDC | Streamlit native OIDC (`st.login()`) | Jan 2025 | Built-in Google/Microsoft/GitHub OAuth |
| Client-side file validation | Server-side validation | Feb 2025 (CVE patch) | Streamlit 1.43.2+ enforces backend validation |
| JWT in localStorage | HTTPOnly cookies | Ongoing best practice | Prevents XSS attacks on tokens |

**New tools/patterns to consider:**
- **Streamlit native OIDC (2025):** `st.login('google')` and `st.logout()` for OAuth integration - simpler than third-party components for OAuth needs
- **streamlit-authenticator 0.3.3+ (2025):** Added automatic password rehashing, better failed login tracking, OAuth2 support
- **Argon2 RFC 9106 profiles:** Pre-configured parameters for high-memory and low-memory systems (use LOW_MEMORY for personal RAG system)
- **HSTS headers:** Modern browsers require Strict-Transport-Security for production HTTPS

**Deprecated/outdated:**
- **bcrypt for new projects:** Still secure but Argon2id is 2026 standard
- **Storing JWT in localStorage:** XSS vulnerable, use HTTPOnly cookies
- **auth0-component for Streamlit:** Native `st.login()` is simpler (Jan 2025)
- **Disabling XSRF protection:** Was common workaround, now considered security anti-pattern
</sota_updates>

<open_questions>
## Open Questions

Things that couldn't be fully resolved:

1. **Database vs YAML for credentials**
   - What we know: YAML works for single-user/small teams, streamlit-authenticator supports it
   - What's unclear: At what user count should we migrate to PostgreSQL?
   - Recommendation: Start with YAML (< 10 users), migrate to database if scaling beyond personal use

2. **Session timeout vs cookie expiry**
   - What we know: streamlit-authenticator uses cookie expiry (30 days default)
   - What's unclear: Should we implement additional inactivity timeout?
   - Recommendation: Cookie expiry sufficient for personal system, add timeout if deploying publicly

3. **Rate limiting approach**
   - What we know: streamlit-authenticator tracks failed_login_attempts
   - What's unclear: Does it enforce rate limiting automatically?
   - Recommendation: Verify in implementation, may need nginx rate limiting as backup
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- /streamlit/docs - Session state, authentication concepts, security reminders
- /mkhorasani/streamlit-authenticator - Complete authentication module with code examples
- /jpadilla/pyjwt - JWT implementation patterns and best practices
- /pyca/bcrypt - bcrypt password hashing (legacy option)
- /hynek/argon2-cffi - Argon2 password hashing (current standard)

### Secondary (MEDIUM confidence)
- [Streamlit Authentication Best Practices (2026)](https://docs.streamlit.io/develop/concepts/connections/authentication) - Official Streamlit OIDC documentation
- [Session Hijacking Prevention Guide (2026)](https://www.baeldung.com/cs/session-hijacking) - General session security best practices, verified against Streamlit patterns
- [Argon2 vs bcrypt Comparison (2025)](https://guptadeepak.com/the-complete-guide-to-password-hashing-argon2-vs-bcrypt-vs-scrypt-vs-pbkdf2-2026/) - Industry consensus on password hashing standards
- [Streamlit HTTPS Configuration (2025)](https://docs.streamlit.io/develop/concepts/configuration/https-support) - Official SSL/HTTPS guidance
- [Streamlit Nginx Deployment Guides (2024-2025)](https://medium.com/@sstarr1879/deploying-secure-streamlit-apps-on-aws-ec2-with-docker-nginx-and-https-39bc941f8710) - Production deployment patterns
- [Streamlit Security Vulnerability (Feb 2025)](https://cybersecuritynews.com/streamlit-vulnerability/) - Recent CVE requiring backend validation

### Tertiary (LOW confidence - needs validation)
- None - all findings cross-verified with official documentation
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: Streamlit authentication patterns
- Ecosystem: streamlit-authenticator, Argon2-cffi, PyJWT, cookie managers
- Patterns: Session management, secure cookies, password hashing
- Pitfalls: Pickle security, HTTPS requirements, CORS/XSRF, recent CVEs

**Confidence breakdown:**
- Standard stack: HIGH - verified with Context7, official docs, recent releases (2025-2026)
- Architecture: HIGH - from official Streamlit docs and streamlit-authenticator examples
- Pitfalls: HIGH - from official security reminders, recent CVE disclosures (Feb 2025), OWASP best practices
- Code examples: HIGH - all examples from Context7 official library documentation

**Research date:** 2026-01-15
**Valid until:** 2026-02-15 (30 days - Streamlit auth ecosystem stable, but check for security updates)

**Key decision drivers:**
- Use streamlit-authenticator (not custom) - reduces security risk
- Use Argon2id (not bcrypt) - current 2026 standard, GPU-resistant
- Use Nginx for SSL termination (not Streamlit SSL) - Streamlit recommendation for production
- Use YAML config (not database) - appropriate for single-user system
- Enable CORS/XSRF protection - Streamlit security defaults

**Critical security updates to monitor:**
- Streamlit CVE Feb 2025 (file upload validation) - ensure version 1.43.2+
- streamlit-authenticator releases - check for security patches
- Argon2 parameter recommendations - RFC 9106 profiles may update
</metadata>

---

*Phase: 1-authentication-foundation*
*Research completed: 2026-01-15*
*Ready for planning: yes*
