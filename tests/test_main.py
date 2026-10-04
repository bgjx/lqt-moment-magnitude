"""Unit test for main.py"""

import pytest

from lqtmoment import main


class TestMain:
    def test_main_help(self, capsys):
        """Test main() handles help."""
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "Calculate moment magnitude" in captured.out

    def test_main_invalid_args(self, capsys):
        """Test main() rejects invalid args."""
        with pytest.raises(SystemExit) as exc:
            main(["--nonsense"])
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert "unrecognized arguments" in captured.err
