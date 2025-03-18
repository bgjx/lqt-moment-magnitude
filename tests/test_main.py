""" Unit test for main.py """

import pytest
from LQTMomentMag.main import main

def test_main_help(capsys):
    """ Test main() handles help. """
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Calculate moment magnitude" in captured.out


def test_main_invalid_args(capsys):
    """ Test main() rejects invalid args."""
    with pytest.raises(SystemExit) as exc:
        main(["--nonsense"])
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err