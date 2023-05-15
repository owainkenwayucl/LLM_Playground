import messages
import xml.etree.ElementTree as _ElementTree

_DATA_PATH="{http://www.tei-c.org/ns/1.0}text"

def process(filename):
    messages.debug("Processing file: " + str(filename))
    #_process_elementtree(filename)
    _process_splittag(filename)

def _process_splittag(filename):
    data = ""
    with open(filename, "r") as file:
        data = file.read()

    temp = data.split("<body>")[1]
    temp = temp.split("</body>")[0]

    # Easy replacements
    spaces = ["<lb/>"]
    nulls = ["<p>","</p>", "<div>", "</div>"]

    for a in nulls:
        temp = temp.replace(a, "")
    for a in spaces:
        temp = temp.replace(a, " ")

    # Deletions
    while "<del>" in temp:
        parts = temp.split("<del>", 1)
        left = parts[0]
        right = parts[1]
        parts = right.split("</del>", 1)
        right = parts[1]

        temp = left + right



    data = temp
    
    messages.log(data, filename)


def _process_elementtree(filename):   
    try:
        tree = _ElementTree.parse(filename)
        tree_root = tree.getroot()
        data = tree_root.findall(_DATA_PATH)[0]

        output = _ElementTree.tostring(data, encoding="utf-8", method="xml")

        messages.log(output, filename)

    except Exception as e:
        messages.error("XML Error in " + filename)
        messages.error(str(e))