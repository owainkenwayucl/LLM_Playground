# Ollama Docker container notes.

We have a new user of ATLAS who uses Ollama. Ollama provides a container but this only supports Nvidia. Can we build an AMD container?

Plan: Use base container `docker.io/rocm/dev-ubuntu-24.04`