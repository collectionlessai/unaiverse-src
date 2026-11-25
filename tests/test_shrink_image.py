"""The shrink of a picture sent with /send: down to an area (Custom.MAX_IMAGE_PIXELS), never up."""
import pytest
from PIL import Image

from unaiverse.custom import Custom
from unaiverse.networking.node.node import Node

pytestmark = pytest.mark.unit

shrink = Node._Node__shrink_image


def test_a_large_photo_is_shrunk_to_the_area_keeping_its_shape():
    photo = Image.new('RGB', (4000, 3000))

    small = shrink(photo, Custom.MAX_IMAGE_PIXELS['sd'])

    assert small.width * small.height <= Custom.MAX_IMAGE_PIXELS['sd'] * 1.01
    assert small.width * small.height >= Custom.MAX_IMAGE_PIXELS['sd'] * 0.99
    assert abs(small.width / small.height - 4 / 3) < 0.01


def test_hd_keeps_four_times_the_area_of_sd():
    photo = Image.new('RGB', (6000, 4000))

    sd = shrink(photo, Custom.MAX_IMAGE_PIXELS['sd'])
    hd = shrink(photo, Custom.MAX_IMAGE_PIXELS['hd'])

    assert abs((hd.width * hd.height) / (sd.width * sd.height) - 4) < 0.05


def test_a_panorama_weighs_about_as_much_as_a_square_photo():
    panorama = shrink(Image.new('RGB', (12000, 2000)), Custom.MAX_IMAGE_PIXELS['sd'])
    square = shrink(Image.new('RGB', (5000, 5000)), Custom.MAX_IMAGE_PIXELS['sd'])

    assert abs(panorama.width * panorama.height - square.width * square.height) < 0.02 * Custom.MAX_IMAGE_PIXELS['sd']


def test_a_small_picture_keeps_its_size():
    icon = Image.new('RGB', (640, 480))

    assert shrink(icon, Custom.MAX_IMAGE_PIXELS['sd']) is icon
