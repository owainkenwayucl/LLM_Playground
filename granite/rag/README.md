# Tools for doing RAG with Granite.

You can generate an example dataset with the Public Domain works of H.P. Lovecraft by downloading all the URLs in `data_lovecraft_gutenberg.txt` into a `data` subdirectory.

E.g.

```
$ mkdir data
$ cd data
$ wget -i ../data_lovecraft_gutenberg.txt
$ cd ..
$ python3 rag.py
Starting up - embedding model = ibm-granite/granite-embedding-125m-english
Starting up - LLM checkpoint = ibm-granite/granite-3.2-8b-instruct
Detected 1 Cuda devices.
Detected Cuda Device 0: AMD Instinct MI300X
Loading 20 documents from ./data...done.
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.20it/s]
? did danforth enjoy his expedition?
---
🤖 : Based on the provided context, it is not explicitly stated that Danforth enjoyed his expedition. In fact, the text suggests the opposite. Danforth was visibly shaken and traumatized by the horrors he witnessed, as evidenced by his shrieking and subsequent breakdown. His nervous pitch and trembling hands during the flight suggest a high level of fear and distress. Therefore, it is safe to infer that Danforth did not enjoy his expedition.
---

? bye
```