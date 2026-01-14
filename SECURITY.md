# Security Policy

## Reporting a Vulnerability

We take the security of llamaIndex-local-rag seriously. If you have discovered a security vulnerability, please report it to us privately.

### How to Report

**Please do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security issues by:

1. **Email**: Send details to the project maintainers (contact information in README.md)
2. **GitHub Security Advisory**: Use GitHub's [private security advisory feature](https://github.com/YOUR_USERNAME/llamaIndex-local-rag/security/advisories/new)

### What to Include

When reporting a vulnerability, please include:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if you have one)
- Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours of your report
- **Status Update**: Within 7 days with assessment and timeline
- **Resolution**: Varies by severity, typically 30-90 days

### Scope

The following are **in scope** for vulnerability reports:

- Authentication and authorization bypasses
- SQL injection, command injection, XSS
- Remote code execution vulnerabilities
- Exposure of sensitive data (API keys, credentials, personal data)
- Denial of service vulnerabilities
- Security misconfigurations in default settings

The following are typically **out of scope**:

- Vulnerabilities in dependencies (please report to the upstream project)
- Issues that require physical access to a user's device
- Social engineering attacks
- Denial of service requiring massive computational resources

### Disclosure Policy

- Security issues are fixed in private development branches
- Once patched, we'll release a security advisory and new version
- We aim for coordinated disclosure - please allow us time to fix before public disclosure
- We'll credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices for Users

### Protecting Sensitive Data

- **Never commit** `.env` files or credentials to version control
- Store sensitive data in `.env` files (gitignored by default)
- Use strong passwords for PostgreSQL database
- Restrict database access to localhost unless necessary

### Network Security

- PostgreSQL should only accept connections from trusted hosts
- Use SSH tunneling for remote database access
- Configure firewall rules to restrict access to ports 5432, 3000 (Grafana), 9090 (Prometheus)

### Data Privacy

- Chat logs and personal data (Facebook Messenger exports) contain sensitive information
- Ensure `data/` directory is gitignored (it is by default)
- Be cautious when sharing query logs or embeddings
- Consider encryption at rest for highly sensitive document collections

### Docker Security

- Don't run containers as root (configured properly in docker-compose.yml)
- Keep Docker images updated (`docker-compose pull`)
- Review environment variables in docker-compose.yml before deployment
- Use Docker secrets for production deployments

### vLLM / LLM Security

- Local LLMs are isolated but still process user input
- Sanitize user queries before passing to LLM
- Monitor resource usage to prevent DoS via large prompts
- Be aware that LLMs can occasionally output training data

### Monitoring and Auditing

- Review Grafana dashboards for unusual activity
- Monitor query logs for suspicious patterns
- Keep logs for security incident investigation
- Enable audit logging for production environments

## Dependency Security

We use:

- **Dependabot**: Automatically checks for vulnerable dependencies
- **pip-audit**: Scans Python dependencies for known vulnerabilities
- **Ruff**: Includes security linting rules

To check dependencies yourself:

```bash
# Check for known vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt

# Run security audit
make security-check  # (if available)
```

## Security Updates

Security updates are released as:

- **Critical**: Immediate patch release (same day)
- **High**: Patch release within 7 days
- **Medium**: Included in next minor release
- **Low**: Included in next major release

Subscribe to GitHub releases to receive security notifications.

## Security Audits

Recent security audits:

- **2026-01**: Comprehensive security audit completed (see `docs/reports/audits/SECURITY_AUDIT_REPORT.md`)
- **2026-01**: Access control audit (see `docs/reports/audits/SECURITY_AUDIT_ACCESS_CONTROL.md`)

## Contact

For security-related questions (not vulnerability reports):

- **GitHub Discussions**: General security questions
- **GitHub Issues**: Non-sensitive security improvements

For vulnerability reports, use the private reporting methods above.

## Acknowledgments

We thank the following security researchers for responsible disclosure:

- (None yet - be the first!)

---

**Last Updated**: January 2026
