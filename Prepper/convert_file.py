import messages
import xml.etree.ElementTree as _ElementTree
import hashlib
import os

_DATA_PATH="{http://www.tei-c.org/ns/1.0}text"

def process(filename, outputdir="."):
    messages.debug("Processing file: " + str(filename))
    data = ""
    try:
        with open(filename, "r") as file:
            data = file.read()
        sha256 = str(hashlib.sha256(data.encode("UTF-8")).hexdigest())

    
        #output = str(_process_elementtree(data, filename))
        output = _process_splittag(data, filename)
        #output = str(_process_elementtree(output, filename))
        messages.debuglog(output, filename)

        outfile = os.path.join(outputdir, sha256 + ".txt")
        with open(outfile, "w") as file:
            file.write(output)

    except Exception as e:
        messages.error("Error processing " + filename)
        messages.error(str(e))


def _process_splittag(xmlstring, filename):
    data = xmlstring
   
    temp = data.split("<body>")[1]
    temp = temp.split("</body>")[0]

    # Easy replacements
    spaces = ["<lb/>", "</hi>"]
    nulls = ["<p>","</p>", "<div>", "</div>"]
    longs = ["hi"]

    for a in nulls:
        temp = temp.replace(a, "")
    for a in spaces:
        temp = temp.replace(a, " ")
    for a in longs:
        while "<"+a in temp:
            parts = temp.split("<" + a, 1)
            left = parts[0]
            right = parts[1]
            parts = right.split(">", 1)
            right = parts[1]

            temp = left + " " + right

    # Deletions
    while "<del>" in temp:
        parts = temp.split("<del>", 1)
        left = parts[0]
        right = parts[1]
        parts = right.split("</del>", 1)
        right = parts[1]

        temp = left + right

    data = temp

    return data


def _process_elementtree(xmlstring, filename):   
    try:
        tree = _ElementTree.ElementTree(_ElementTree.fromstring(xmlstring))
        tree_root = tree.getroot()
        data = tree_root.findall(_DATA_PATH)[0]

        output = _ElementTree.tostring(data, encoding="utf-8", method="xml")

        return output

    except Exception as e:
        messages.error("XML Error in " + filename)
        messages.error(str(e))
        return ""