#!/usr/bin/env python3
# This script re-arranges data from the Transcribe Bentham dataset into a format that Dolly can use.
# This script walks a directory and processes each file.
import messages
import convert_file

def _main(dirname):
    import os
    for root, dirs, files in os.walk(str(dirname)):
        for file in files:
            if file.endswith(".xml"):
                filename = os.path.join(root, file)
                convert_file.process(filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Convertor for Transcribe Bentham XML data")
    parser.add_argument("-d", action="store_true", help="Turn on Debug mode.", default=False)
    parser.add_argument("directory", metavar="directory", type=str, nargs="+", help="Directory tree of files to process.")
    args = parser.parse_args()
	
    messages.DEBUG=args.d 

    for a in args.directory:
        _main(a)

