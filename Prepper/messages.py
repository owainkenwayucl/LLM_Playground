import sys

DEBUG=True

def debug(message):
    if (DEBUG):
        sys.stderr.write(">>> DEBUG <<< : " + str(message) +"\n")

def error(message):
    sys.stderr.write(">>> ERROR <<< : " + str(message) + "\n")

def log(metadata="LOG message", message):
    print(" >>> Begin " + str(metadata) + " <<<" )
    print(str(message))
    print(" >>> End " + str(metadata) + " <<<" )