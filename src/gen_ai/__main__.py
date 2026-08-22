import logging

from gen_ai.main import main

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logging.exception("Project error: %s", error)
        raise SystemExit(1) from error
