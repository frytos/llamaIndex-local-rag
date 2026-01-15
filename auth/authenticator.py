"""Authentication module initialization for Streamlit app."""
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate
from typing import Tuple

def load_authenticator() -> Authenticate:
    """Load and initialize Streamlit authenticator from config.

    Returns:
        Authenticate: Configured authenticator instance
    """
    with open('auth/config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    return Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
