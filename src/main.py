import asyncio
from viam.module.module import Module
from models.vision import Vision as VisionModel


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
