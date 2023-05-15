import messages
import xml.etree.ElementTree as ElementTree

_DATA_PATH="{http://www.tei-c.org/ns/1.0}text"

def process(filename):
    messages.debug("Processing file: " + str(filename))

    try:
        tree = ElementTree.parse(filename)
        tree_root = tree.getroot()
        data = tree_root.findall(_DATA_PATH)[0]

        output = ElementTree.tostring(data, encoding="utf-8", method="xml")

        messages.log(output, filename)

    except Exception as e:
        messages.error("XML Error in " + filename)
        messages.error(str(e))
