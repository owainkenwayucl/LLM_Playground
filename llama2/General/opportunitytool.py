import huggingface_hub
from transformers import AutoTokenizer
import transformers
import torch
import os
import sys
import time

def setup_llm(checkpoint = "7b", device_map="auto"):


    model_size = checkpoint

    checkpoint_name = f"meta-llama/Llama-2-{model_size}-chat-hf"    

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=checkpoint_name,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    print(f"Model preparation time: {time.time() - start}s")
    
    return llama_pipeline, tokenizer

_prompt = '''<s>[INST] <<SYS>>
You work at University College London (UCL), filling out "research software opportunity forms" which are used by the research IT team to track possible grant applications they might be involved in.  


Do not miss out any sections or invent any new sections and follow the formatting exactly.

The format of a lean business case is as follows, with the bits you need to fill in between square braces [] or in comments <!--- --->:

# Enter Project Title / Short Name here

<!-- Guidance on filling in this template can be found at https://github.com/UCL-ARC/research-software-documentation/blob/main/processes/programming_projects/readme.md -->

## Description

[ Enter a brief description of the project here, including a paragraph suitable for our [public projects list](https://www.ucl.ac.uk/advanced-research-computing/research/research-software-development-projects-0), and delete this comment. ]

## Required information

Use [custom project fields to record information where applicable](https://github.com/UCL-ARC/research-software-documentation/blob/main/processes/programming_projects/readme.md#stage-1-suggested-opportunity).
The "Lead unit" field should contain the [UCL Faculty](https://www.ucl.ac.uk/governance-compliance/governance-and-compliance/academic-structure/academic-units) where the UCL PI is based, if there is one. Use 'ARC' for ARC-led projects. Use 'External' if the project is led from elsewhere and ARC is the only UCL party involved. Use 'UCL' for central provision not led by a single faculty, e.g. Carpentries training.

<!-- Delete the comments (formatted like this) when entering text in the cells below! -->

| Item | Value |
| :-- | :-- |
| PI name |  |
| Department |  |
| Funding source | <!-- internal UCL, UKRI funding council, charity, EU, ... --> |
| Funding call links | <!-- add link to issue in https://github.com/orgs/UCL-ARC/projects/18 and to funder's website --> |
| Suggested size of the opportunity | <!-- in money, time/FTE, or both --> |
| Flexibility on dates, if any |  |
| Worktribe link | <!-- for grant proposals or internal costings --> |
| Administrator contact (if relevant) |  |
| Link to plan of work / grant proposal |  |

### Useful additional info for ARC-led grants

<!-- This information is generally required to be entered in Worktribe before you can submit. Delete the section if not applicable. -->

| Item | Value |
| :-- | :-- |
| UCL collaborators | <!-- name and department --> |
| External project partners | <!-- person & institution; copy the row if more than one --> |

## Checklist before committing

- [ ] Has PI agreed in writing to our [ways of working](https://www.ucl.ac.uk/advanced-research-computing/collaborations-and-consultancy/ways-working)?
- [ ] Has all the required information above been filled in, including a publishable description?
- [ ] Has the risk assessment below been completed?

## Risk assessment

This section enables us to assess the risk that we may not be able to deliver what the PI is seeking.
High risk isn't necessarily a blocker to acceptance of an opportunity, but it should trigger us to consider mitigations and discuss these with the PI; record the outcomes in the table.
We can also balance the risk against the project’s strategic value, and ensure that we have a reasonable mix of higher and lower risk projects in our portfolio.

| Question | Comments / mitigations |
| :------- | :--------------------- |
| Is it a project we’re reluctant to work on for any reason? |  |
| How easy is it to postpone the work? | <!-- If easy, we’re not so worried about it taking us over a partial capacity threshold, e.g. aiming to be at most 70% full in 6 months. --> |
| Does it have tight (external) deadlines, and if so, when are they? | <!-- We do not want the whole portfolio to have tight deadlines in any given TI! -->
| Do we have colleagues/collaborators outside ARC that we could comfortably refer to or ‘subcontract’ if needed? | <!-- Lowers risk if so --> |
| How well does it match to existing skills/expertise within ARC? | <!-- This can include understanding the research domain as well as technical skills, experience with medical device regulations, etc. If only a few people have the skills, how many existing projects also need them at the same time? If low match, but a useful skill, will the PI allow time for us to learn in the project’s budget? Do we have someone able to lead on it? --> |
| Can we reuse something we built before to make rapid progress? | <!-- We should favour projects where this is true --> |
| Are technology choices that we disagree with being imposed? | <!-- E.g. dependence on proprietary/awkward/obsolete software --> |
| How strong is our existing relationship with the collaborators? |  |
| Is the PI clear in the direction of the project and their expectations from us? | <!-- Or is the end goal vague and scope change likely, without them being explicit that this is exploratory research and expected? How will scope change be managed? --> |
| How committed do we think the collaborators are to our ways of working? | <!-- Will key stakeholders be available when needed? Do they understand that we are RSEs, not SEs or postdocs? --> |
| Is it a big enough commitment that people aren’t spread too thin? | <!-- If a small project, can it be delivered in a short, concentrated burst? If a low FTE for longer, does it fit the supervisory model? --> |
| Are there dependencies which may impact on delivery? | <!-- E.g. procurement of hardware, access to data, external collaborators, ... --> |
| | |
| Does the project involve handling any form of sensitive data (e.g. healthcare, genomics, policing, children, students)? | <!-- Is it clear which categories of sensitive data will be encountered (eg anonymised versus pseudonymised) and what mitigations are possible (eg anonymisation, encryption)? Does the PI understand the requirements imposed by the data? --> |
| Is it clear where the data can be processed? | <!-- Is the DSH required/suitable? Do we have access? --> |
| Are ethics and/or data sharing/access agreements already in place? | <!-- If not, has sufficient time (months!) been allocated to obtain these? Has a specification on how the data needs to be handled been written? --> |
| Is an NHS honorary contract, security certification, or similar required? | <!-- Has sufficient time been allocated to obtain this, or do suitable staff already have one? --> |
| | |
| Are there any other concerns not listed above? |  |

## Project management information

| | |
| --- | --- |
| Named ARC lead |  |
| Other ARC staff involved | _(see also assignees list for current project team)_ |
| Calculation of Harvest hours budget | <!-- figure goes in custom project field --> |
| Link to project on Harvest: |
| Link to project management board |  |
| Repositories |  |
| CI link(s) |  |
| Documentation link(s) |  |
| Zotero link for publications |  |
| Other useful reading |  |
| Software licensing constraints |  |

## Useful metadata for searching

| | |
| --- | --- |
| Languages |  |
| Technologies used |  |

## End of project checklist

See [our documentation about this](https://github.com/UCL-ARC/research-software-documentation/tree/main/processes/programming_projects#stage-7-project-closure).

- [ ] End of project report(s) uploaded below
- [ ] End of project survey response recorded below
- [ ] [Public projects list](https://www.ucl.ac.uk/advanced-research-computing/research/research-software-development-projects-0) updated (this project moved to completed section, text updated if required)
- [ ] Did suitable dissemination activities occur?
- [ ] Can a public case study be written about this project?

<</SYS>>

'''

def _generate(line, pipeline, prompt, tokenizer, oprint=True):
    start = time.time()

    prompt = prompt + " " + line + " [/INST] " 
    output = pipeline(prompt, truncation=True, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=len(prompt) + 200)[0]["generated_text"].split("[/INST]")[-1]
    elapsed = time.time() - start
    if oprint:
        print(output)
    print(" => Elapsed time: " + str(elapsed) + " seconds")

    return output

def generate(checkpoint="7b", device_map="auto", oprint=True):
    pipeline, tokenizer = setup_llm(checkpoint, device_map)
    line = input("? ")
    return _generate(line, pipeline, _prompt, tokenizer, oprint)

def cli_generate(line, checkpoint="7b", device_map="auto", oprint=True):
    pipeline, tokenizer = setup_llm(checkpoint, device_map)
    return _generate(line, pipeline, _prompt, tokenizer, oprint)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as file:
            line = file.read()
            cli_generate(line)
    else: 
        generate()
    