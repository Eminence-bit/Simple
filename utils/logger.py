import logging

def setup_logger(name):
    # Clear any existing handlers
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers = []
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler with detailed formatting
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Prevent log propagation to avoid duplicate messages
    logger.propagate = False


    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger