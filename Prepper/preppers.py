#!/usr/bin/env python3
# This script re-arranges data from the Transcribe Bentham dataset into a format that Dolly can use.
# This script walks a directory and processes each file.
import messages
import convert_file

def _main(dirname, outdir, processes):
    if processes <= 1:
        _serial_execute(_generate_file_list(dirname), outdir)
    else:
        messages.debug("Enabling Parallel mode with " + str(processes) + " processes.")
        _parallel_multiprocessing_execute(_generate_file_list(dirname), outdir, processes)

def _generate_file_list(dirname):
    import os
    fileslist = []
    messages.debug("Generating File list...")
    for root, dirs, files in os.walk(str(dirname)):
        for file in files:
            if file.endswith(".xml"):
                filename = os.path.join(root, file)
                fileslist.append(filename)
    messages.debug("File list generated.")
    return fileslist

def _serial_execute(files, outdir):
    for a in files:
        convert_file.process(a, outdir)

def _parallel_multiprocessing_execute(files, outdir, processes):
    l = len(files)
    message.debug("Files detected: " + str(l))
    # fix edge case
    if l < processes:
        messages.debug("More processes than files, setting processes to " + str(l))
        processes = l

    chunksize = int(l/processes)
    remainder = l % processes
    if remainder > 0:
        chunksize += 1

    chunkedlist = [files[i:i+chunksize] for i in range(0, l, chunksize)]

    clsize = []
    for a in chunkedlist:
        clsize.append(len(a))
    message.debug("List sizes: " + str(clsize) + " for " + sum(clsize) + " total files.")
    messages.error("Not implemented")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Convertor for Transcribe Bentham XML data")
    parser.add_argument("-D", action="store_true", help="Turn on Debug mode.", default=False)
    parser.add_argument("-o", metavar="outdir", type=str, help="Output folder for processed files.", default=".")
    parser.add_argument("-p", metavar="processes", type=int, help="Number of processes to use", default=1)
    parser.add_argument("directory", metavar="directory", type=str, nargs="+", help="Directory tree of files to process.")
    args = parser.parse_args()
	
    messages.DEBUG=args.D
    
    for a in args.directory:
        _main(a, args.o, args.p)

