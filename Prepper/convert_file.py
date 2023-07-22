import messages
import xml.etree.ElementTree as _ElementTree
import hashlib
import os
import sys

_DATA_PATH="{http://www.tei-c.org/ns/1.0}text"
_SHEEP_CONFIGURED = False
_LLM_ENDPOINT = None
_PLATFORM_GRAPHCORE = False
_PROMPT = "You are an AI tasked with processing XML data and converting it into plain text by interpreting tags. As an AI tasked with this difficult job, you are professional and not chatty. Please process up the following XML snippet: "
#_MODEL = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
_MODEL = "databricks/dolly-v2-12b"

# Modes
# 1. Split the xml string on <body> tags and then manually strip known tags (incomplete)
# 2. Use ElementTree.tostring to remove all tags (loses meaning)
# 3. Hybrid uses a mix of 1. and 2. (should be best?) 
MODES = ["split", "elementtree", "hybrid", "beautifulsoup", "sheep"]

def process(filename, outputdir=".", mode="split"):
	sys.argv = [sys.argv[0]] # fix weird poplar bug
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
		elif mode == "beautifulsoup":
			output = _process_beautifulsoup(data, filename)
		elif mode == "sheep":
			output = _process_sheep(data, filename)
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
	data = str(xmlstring)
   
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

def _process_sheep(xmlstring, filename):
	if not _SHEEP_CONFIGURED:
		_setup_sheep()
	if _PLATFORM_GRAPHCORE:
		return _LLM_ENDPOINT(_PROMPT + xmlstring)[0]
	else:
		return _LLM_ENDPOINT(_PROMPT + xmlstring)[0]["generated_text"]

def _setup_sheep():
	messages.debug("Setting up LLM toolchain")
	import torch
	if _PLATFORM_GRAPHCORE:
		messages.debug("Graphcore environment detected.")
		number_of_ipus = int(os.getenv("NUM_AVAILABLE_IPU", 16))
		number_of_ipus

		from utils.setup import dolly_config_setup

		config_name = "dolly_pod4" if number_of_ipus == 4 else "dolly_pod16"
		config, *_ = dolly_config_setup("config/inference.yml", "release", config_name)
		config

		import api

		sequence_length = 1024
		micro_batch_size = 4

		pipeline = api.DollyPipeline(config, sequence_length=sequence_length, micro_batch_size=micro_batch_size, hf_dolly_checkpoint=_MODEL)
	else:		
		from instruct_pipeline import InstructionTextGenerationPipeline
		from transformers import AutoModelForCausalLM, AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained(_MODEL, padding_side="left")
		model = AutoModelForCausalLM.from_pretrained(_MODEL, device_map="auto", torch_dtype=torch.bfloat16)
		pipeline = InstructionTextGenerationPipeline(model=model, task="text-generation", tokenizer=tokenizer, return_full_text=False)

	global _LLM_ENDPOINT
	global _SHEEP_CONFIGURED
	_LLM_ENDPOINT = pipeline
	_SHEEP_CONFIGURED = True
	messages.debug("Setup Complete.")

def _process_beautifulsoup(xmlstring, filename):
	try:
		from bs4 import BeautifulSoup
		doc_soup = BeautifulSoup(xmlstring, features="xml")
		body_text = str(doc_soup.find_all("body")[0])
		body_soup = BeautifulSoup(body_text, features="xml") # horrible
		output = body_soup.get_text()
		return output

	except Exception as e:
		messages.error("XML Error in " + filename)
		messages.error(str(e))
		messages.debug(xmlstring)
		return ""