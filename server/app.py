"""Server entry point for OpenEnv validation."""
import uvicorn
from app import app

def main():
    """Entry point for the server script."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
