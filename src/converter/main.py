from __future__ import annotations

from dotenv import load_dotenv
import uvicorn

from .api import create_app
from .config import Settings


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
