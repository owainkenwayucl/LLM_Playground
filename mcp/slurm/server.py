from mcp.server import MCPServer
import asyncio

def run(command, stdin=None, cwd=None):
	import subprocess
	return subprocess.run(command, input=stdin, cwd=cwd, capture_output=True, encoding='UTF-8')

mcp = MCPServer("Slurm")

@mcp.resource("clusterinfo://")
def clusterinfo() -> str:
	command = ["sinfo", "--json"]
	return run(command).stdout

@mcp.resource("clusterqueue://")
def clusterqueue() -> str:
	command = ["squeue", "--json"]
	return run(command).stdout

@mcp.tool()
def submitjob(jobscript: str, directory: str) -> str:
	command = ["sbatch"]
	return run(command, stdin=jobscript, cwd=directory).stdout

@mcp.tool()
def submitjobscriptfile(file: str, directory: str) -> str:
	command = ["sbatch", file]
	return run(command, cwd=directory).stdout

@mcp.tool()
def canceljob(jobid: int):
	command = ["scancel", str(jobid)]
	run(command)

@mcp.tool()
def jobstatus(jobid: int) -> str:
	command = ["squeue", "--json", "-j", str(jobid)]
	return run(command).stdout

@mcp.tool()
def clusterrun(command: str, directory: str) -> str:
	command_s = command.split()
	command_a = ["srun"]
	for a in command_s:
		command_a.append(a)
	return run(command_a, cwd=directory).stdout

# Run through stdio
if __name__ == "__main__":
	asyncio.run(mcp.run(transport="stdio"))
