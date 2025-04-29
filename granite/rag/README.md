# Tools for doing RAG with Granite.

You can generate an example dataset with the Public Domain works of H.P. Lovecraft by downloading all the URLs in `data_lovecraft_gutenberg.txt` into a `data` subdirectory.

E.g.

```
mkdir data
cd data
wget -i ../data_lovecraft_gutenberg.txt
cd ..
python3 rag.py
```