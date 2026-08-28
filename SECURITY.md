# Security policy

## Reporting

Use a private GitHub security advisory where available. Include the affected commit, minimal reproduction, impact, preconditions and remediation proposal. Do not attach real learner recordings, access tokens, API keys or personal transcripts.

## Security boundaries

- Local LLM endpoints are loopback HTTP only.
- Environment proxies are disabled for teacher requests.
- HTTP redirects are rejected.
- Candidate IDs must exactly match the request set.
- Rank-only teachers cannot create observed text.
- Cache entries are canonicalized and SHA-256 verified.
- Unknown cache schemas fail closed.
- Raw audio and model weights are forbidden from the repository.
- Exported transcript metadata omits absolute source paths.

Reports demonstrating a bypass of any boundary are especially valuable.
