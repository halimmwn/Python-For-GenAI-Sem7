import os
from dotenv import load_dotenv

load_dotenv() # loads .env into os.environ


def get_api_key(provider: str) -> str:
	"""Retrieve an API key from the environment."""
	key_map = {
		"anthropic": "ANTHROPIC_API_KEY",
		"openai": "OPENAI_API_KEY",
		"google": "GOOGLE_API_KEY",
	}
	env_var = key_map.get(provider.lower())
	if not env_var:
		raise ValueError(f"Unknown provider:{provider}")
	key = os.getenv(env_var)
	if not key:
		raise EnvironmentError(
			f"{env_var} is not set. Add it to your .env file."
		)
	return key


# In practice:
# anthropic_key = get_api_key("anthropic")
# openai_key = get_api_key("openai")


if __name__ == "__main__":
	print("Environment variables loaded successfully")
	for provider, env_var in {
		"anthropic": "ANTHROPIC_API_KEY",
		"openai": "OPENAI_API_KEY",
		"google": "GOOGLE_API_KEY",
	}.items():
		status = "available" if os.getenv(env_var) else "not configured"
		print(f"{provider}: {status}")