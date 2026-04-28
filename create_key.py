import secrets
import string

def generate_api_key(length: int = 48, prefix: str = "sk_") -> str:
    """Generate a secure, random API key (url-safe, cryptographically strong)."""
    alphabet = string.ascii_letters + string.digits + "-_"
    random_part = ''.join(secrets.choice(alphabet) for _ in range(length - len(prefix)))
    return prefix + random_part

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    
    p.add_argument("--length", type=int, default=48)
    p.add_argument("--prefix", type=str, default="sk")

    args = p.parse_args()

    print(generate_api_key(args.length))