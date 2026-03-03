"""Entry point — run MasteryOS via `python -m src`."""

import uvicorn
from src.ui import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
