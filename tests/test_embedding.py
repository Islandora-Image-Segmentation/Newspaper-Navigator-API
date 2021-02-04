from embedding import generate_embeddings
from PIL import Image
import os

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to test_assets
TEST_ASSETS = os.path.join(CURRENT_SCRIPT_DIR, "test_assets")

"""First embedding function test"""

def test_embedding_one():
  #Load first Image
  img = Image.open(os.path.join(TEST_ASSETS, "test_image_one.png")).convert('RGB')

  #Call generate_embeddings from embedding.py
  embedding = generate_embeddings(img)

  #Check to see if function returns non-empty list
  assert len(embedding) > 0

"""Second embedding function test"""

def test_embedding_two():
  #Load first Image
  img = Image.open(os.path.join(TEST_ASSETS, "test_image_two.png")).convert('RGB')

  #Call generate_embeddings from embedding.py
  embedding = generate_embeddings(img)

  # Check to see if function returns non-empty list
  assert len(embedding) > 0
