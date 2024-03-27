# This class defines what a Document Chunk looks like.
# A Document Chunk has:
# 1. A Title/Question
# 2. A context which could be None or could be str context.
# 3. Either an instruction 
#     or
#    An ordered list of sub Document Chunks

class document_chunk:

    def __init__(self, heading, context=None, contents=""):
        self.heading = heading
        self.context = context
        self.contents = contents
        
    