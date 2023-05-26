import messages
import xml.etree.ElementTree as _ElementTree
import hashlib
import os

_DATA_PATH="{http://www.tei-c.org/ns/1.0}text"

# Modes
# 1. Split the xml string on <body> tags and then manually strip known tags (incomplete)
# 2. Use ElementTree.tostring to remove all tags (loses meaning)
# 3. Hybrid uses a mix of 1. and 2. (should be best?) 
MODES = ["split", "elementtree", "hybrid"]

def process(filename, outputdir=".", mode="split"):
	messages.debug("Processing file: " + str(filename))
	data = ""
	try:
		with open(filename, "r") as file:
			data = file.read()
		sha256 = str(hashlib.sha256(data.encode("UTF-8")).hexdigest())

		if mode == "split":
			output = _process_splittag(data, filename)
		elif mode == "elementtree":
			output = _process_elementtree(data, filename)
		elif mode == "hybrid":
			output = _process_hybrid(data, filename)
		else:
			messages.error("Invalid process method: " + mode)
		messages.debuglog(output, filename)

		outfile = os.path.join(outputdir, sha256 + ".txt")
		with open(outfile, "w") as file:
			file.write(output)

	except Exception as e:
		messages.error("Error processing " + filename)
		messages.error(str(e))

# This routine combines manual tag stripping with element tree stripping
def _process_hybrid(xmlstring, filename):
	stripped = _process_strip_named_tags(xmlstring, filename)
	return _process_elementtree(stripped, filename)

# This routine strips out known tags.
def _process_strip_named_tags(xmlstring, filename):
	temp = xmlstring

	# Easy replacements
	spaces = ["<lb/>", "</hi>", "<sic>", "</sic>","<add>","</add>","<unclear>","</unclear>","<foreign>","</foreign>"]
	nulls = ["<p>","</p>", "<div>", "</div>", "<note>", "</note>","<head>","</head>", "<pb/>", "<gap/>"]
	longs = ["hi", "p", "div"]

	for a in nulls:
		temp = temp.replace(a, "")
	for a in spaces:
		temp = temp.replace(a, " ")
	for a in longs:
		while "<"+ a + " " in temp:
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

	return temp

# This routine plits the string on body tags and then uses the manual string stripping
def _process_splittag(xmlstring, filename):
	data = xmlstring
   
	temp = data.split("<body>")[1]
	temp = temp.split("</body>")[0]

	return _process_strip_named_tags(temp, filename)

# This routine abuses elementree.tostring to remove remaining tags.
def _process_elementtree(xmlstring, filename):   
	try:
		tree = _ElementTree.ElementTree(_ElementTree.fromstring(xmlstring))
		tree_root = tree.getroot()
		data = tree_root.findall(_DATA_PATH)[0]

		output = _ElementTree.tostring(data, encoding="utf-8", method="text")

		return output.decode("utf-8")

	except Exception as e:
		messages.error("XML Error in " + filename)
		messages.error(str(e))
		messages.debug(xmlstring)
		return ""