# The Sheep

The sheep is a simple program to call the open source Databricks Dolly v2.0 LLM.

To run on a UCL HPC system, e.g. Myriad, `git clone` the Dolly repo: https://github.com/databrickslabs/dolly.git

Load the `python/3.9.10` environment module (others may work, this is what it was developed with).

Create and instatiate a virtual environment.

Use `pip` to install both the `requirements.txt` and `requirements-dev.txt`

Run `thesheep.py`

## Note:

This has been tested on:
* Myriad (GPU + CPU) + Kathleen (CPU) with python/3.9.10
* AWS Graviton 2 with Ubuntu 22.04 (there seems to be little to no threading happening on this platform?).
