#!/usr/bin/env python3
# This script re-arranges data from the Transcribe Bentham dataset into a format that Dolly can use.

import messages
import convert_file

def _main(filename, outdir):
    convert_file.process(filename, outdir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Convertor for Transcribe Bentham XML data")
    parser.add_argument("-D", action="store_true", help="Turn on Debug mode.", default=False)
    parser.add_argument("-o", metavar="outdir", tpye=str, help="Output folder for processed files.", default=".")
    parser.add_argument("filename", metavar="filename", type=str, nargs="+", help="Files to process.")
    args = parser.parse_args()
	
    messages.DEBUG=args.D

    for a in args.filename:
        _main(a, args.o)

