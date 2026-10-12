"""Exec-form Docker entry point, honoring the platform PORT setting."""

import logging
import os
import uvicorn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run("housing.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")),
                workers=1, access_log=False)
