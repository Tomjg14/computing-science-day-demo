import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # torch and opencv both ship OpenMP on Windows

from .app import main

main()
