from io import BytesIO

from PIL import Image
import pytest


@pytest.fixture
def make_png():
    def create(width: int = 2, height: int = 2) -> bytes:
        output = BytesIO()
        Image.new("RGB", (width, height), (20, 80, 160)).save(
            output,
            format="PNG",
        )
        return output.getvalue()

    return create
