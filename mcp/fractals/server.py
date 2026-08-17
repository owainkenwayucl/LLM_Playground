from mcp.server import MCPServer
import asyncio
import fractals
from mcp.types import ImageContent

mcp = MCPServer("Fractals")

@mcp.tool(structured_output=False)
def mandelbrot(width: int, height: int, xmin: float, xmax: float, ymin: float, ymax: float, max_iter: int) -> list[ImageContent]:
	image = fractals.write_image_matplotlib_base64(fractals.generate_fractal(width, height, fractals.mandel, xmin, xmax, ymin, ymax, max_iter))
	
	return [
		ImageContent(
			type="image",
			data=image,
			mimeType="image/png"
		)
	]

@mcp.tool(structured_output=False)
def julia(width: int, height: int, xmin: float, xmax: float, ymin: float, ymax: float, max_iter: int, c: float, n: int) -> list[ImageContent]:
	j = fractals.generate_julia(c,n)
	image = fractals.write_image_matplotlib_base64(fractals.generate_fractal(width, height, j, xmin, xmax, ymin, ymax, max_iter))
	
	return [
		ImageContent(
			type="image",
			data=image,
			mimeType="image/png"
		)
	]

if __name__ == "__main__":
	asyncio.run(mcp.run(transport="stdio"))
