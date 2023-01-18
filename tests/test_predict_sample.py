import pytest
from src.clip_interrogator.predict_sample import create_source_list
from unittest.mock import patch


def test_create_source_list():
    with patch( "sys.argv", [ "program", "--source_dir", "." ] ):
        assert create_source_list(".") == []
