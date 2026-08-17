from mcp.server import MCPServer
import asyncio
import fractals
from mcp.types import ImageContent, TextContent

mcp = MCPServer("Fractals")

@mcp.tool(structured_output=False)
def mandelbrot(width: int, height: int, xmin: float, xmax: float, ymin: float, ymax: float, max_iter: int) -> list[ImageContent | TextContent]
	image = fractals.write_image_base64(fractals.generate_fractal(width, height, fractals.mandel, xmin, xmax, ymin, ymax, max_iter))
	
	return [
		ImageContent(
			type="image",
			data=image,
			mimeType="image/png",
		)
		TextContent(type="text", text=f"Generated a Mandelbrot set: width: {width}, height: {height}, xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}, max_iter: {max_iter}"), 
	]

@mcp.tool(structured_output=False)
def julia(width: int, height: int, xmin: float, xmax: float, ymin: float, ymax: float, max_iter: int, c: float, n: int) -> list[ImageContent | TextContent]:
	j = fractals.generate_julia(c,n)
	image = fractals.write_image_base64(fractals.generate_fractal(width, height, j, xmin, xmax, ymin, ymax, max_iter))
	
	return [
		ImageContent(
			type="image",
			data=image,
			mimeType="image/png"
		TextContent(type="text", text=f"Generated a Julia set: width: {width}, height: {height}, xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}, max_iter: {max_iter}, c: {c}, n: {n}"), 
		)
	]

if __name__ == "__main__":
	# asyncio.run(mcp.run(transport="streamable-http", host="0.0.0.0", port=8000))
	asyncio.run(mcp.run(transport="stdio"))
