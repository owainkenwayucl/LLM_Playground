#!/usr/bin/env python3
# This script re-arranges data from the Transcribe Bentham dataset into a format that Dolly can use.
# This script walks a directory and processes each file.
import messages
import convert_file

def _main(dirname):
    _serial_execute(_generate_file_list(dirname))

def _generate_file_list(dirname):
    import os
    files = []
    for root, dirs, files in os.walk(str(dirname)):
        for file in files:
            if file.endswith(".xml"):
                filename = os.path.join(root, file)
                files.append(filename)
    return files

def _serial_execute(files):
    for a in files:
        convert_file.process(a)

def _parallel_multiprocessing_execute():
    messages.error("Not implemented")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Convertor for Transcribe Bentham XML data")
    parser.add_argument("-D", action="store_true", help="Turn on Debug mode.", default=False)
    parser.add_argument("directory", metavar="directory", type=str, nargs="+", help="Directory tree of files to process.")
    args = parser.parse_args()
	
    messages.DEBUG=args.D

    for a in args.directory:
        _main(a)

