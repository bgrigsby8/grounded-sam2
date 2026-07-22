import sys
from pathlib import Path

# Make src/ importable so pipeline.py and pointcloud.py can be tested
# standalone, exactly as the module process imports them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
