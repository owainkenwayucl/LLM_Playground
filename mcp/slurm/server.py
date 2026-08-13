from mcp.server import MCPServer
import asyncio

def run(command):
	import subprocess
	return subprocess.run(command, capture_output=True, encoding='UTF-8')

mcp = MCPServer("Slurm")

@mcp.resource("clusterinfo://")
def cluster_info() -> str:
	command = ["sinfo", "--json"]
	return run(command).stdout

@mcp.resource("clusterqueue://")
def clusterqueue() -> str:
	command = ["squeue", "--json"]
	return run(command).stdout

@mcp.tool()
def submitjob(jobscript: str) -> str:
	command = ["sbatch", jobscript]
	return run(command).stdout

@mcp.tool()
def canceljob(jobid: int):
	command = ["scancel", str(jobid)]
	run(command)

@mcp.tool()
def jobstatus(jobid: int) -> str:
	command = ["squeue", "--json", "-j", jobid]
	return run(command).stdout

@mcp.tool()
def clusterrun(command: str) -> str:
	command_s = command.split()
	command_a = ["srun"]
	for a in command_s:
		command_a.append(a)
	return run(command_a).stdout

# Run through stdio
if __name__ == "__main__":
	asyncio.run(mcp.run(transport="stdio"))
