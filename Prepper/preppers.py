#!/usr/bin/env python3
# This script re-arranges data from the Transcribe Bentham dataset into a format that Dolly can use.
# This script walks a directory and processes each file.
import messages
import convert_file

_FILE_PREFIX="JB_"

def _main(dirname, outdir, processes, mode):
    if processes <= 1:
        _serial_execute(_generate_file_list(dirname), outdir, mode)
    else:
        messages.debug("Enabling Parallel mode with " + str(processes) + " processes.")
        _parallel_multiprocessing_execute(_generate_file_list(dirname), outdir, processes, mode)

def _generate_file_list(dirname):
    import os
    fileslist = []
    messages.debug("Generating File list...")
    for root, dirs, files in os.walk(str(dirname)):
        for file in files:
            if file.endswith(".xml") and file.startswith(_FILE_PREFIX):
                filename = os.path.join(root, file)
                fileslist.append(filename)
    messages.debug("File list generated.")
    return fileslist

def _serial_execute(files, outdir, mode):
    for a in files:
        convert_file.process(a, outdir, mode)

def _parallel_multiprocessing_execute(files, outdir, processes, mode):
    from multiprocessing import Process, Queue, cpu_count
    l = len(files)
    messages.debug("Files detected: " + str(l))
    # fix edge case
    if processes > cpu_count():
        messages.debug("More processes than cores, setting processes to " + str(cpu_count()))
        processes = cpu_count()
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
    messages.debug("List sizes: " + str(clsize) + " for " + str(sum(clsize)) + " total files.")

    q = Queue()
    procs = []

    for a in range(processes):
        procs.append(Process(target=_serial_execute, args=(chunkedlist[a],outdir,mode)))
        procs[a].start()


    for a in range(processes):
        procs[a].join()

    #messages.error("Not implemented")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Convertor for Transcribe Bentham XML data")
    parser.add_argument("-D", action="store_true", help="Turn on Debug mode.", default=False)
    parser.add_argument("-o", metavar="outdir", type=str, help="Output folder for processed files.", default=".")
    parser.add_argument("-p", metavar="processes", type=int, help="Number of processes to use", default=1)
    parser.add_argument("-m", metavar="mode", type=str, help="Processing routine", default=convert_file.MODES[0], choices=convert_file.MODES)
    parser.add_argument("directory", metavar="directory", type=str, nargs="+", help="Directory tree of files to process.")
    args = parser.parse_args()
	
    messages.DEBUG=args.D
    
    for a in args.directory:
        _main(a, args.o, args.p, args.m)

