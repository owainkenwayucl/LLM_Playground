import sys

DEBUG=True

def debug(message):
    if (DEBUG):
        print(">>> DEBUG <<< : " + str(message))

def error(message):
    sys.stderr.write(">>> ERROR <<< : " + str(message) + "\n")