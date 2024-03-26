# Notes:

1. Clone the GraphCore Gradient/HuggingFace example repo (https://github.com/graphcore/Gradient-HuggingFace)
2. Set up a virtualenv.
3. Make sure you've logged into HuggingFace through the cli and have license for llama2 attached to your account.
4. Run `owain-llama.py`


On Mandelbrot

```
source ~/Setup/poplar.sh
source ~/Applications/LLAMA2/bin/activate
export PYTHONPATH=$HOME/Source/Gradient-HuggingFace/llama2-chatbot:$PYTHONPATH
```