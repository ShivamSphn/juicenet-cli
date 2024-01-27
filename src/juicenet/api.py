from pathlib import Path
from typing import Any, Union

from .main import main
from .types import JuicenetOutput, StrPath

__all__ = [
    "juicenet",
]


def juicenet(
    path: StrPath,
    /,
    *,
    config: Union[StrPath, dict[str, Any]],
    is_public: bool = False,
    debug: bool = False,
) -> JuicenetOutput:
    """
    Upload a file to usenet

    Parameters
    ----------
    path : str or pathlib.Path
        The path to an existing file. This can either be a string representing the path or a pathlib.Path object.
    config : str, dict, or pathlib.Path
        The configuration to use when processing the file or directory.
        This can either be a dictionary, a string representing the path to a configuration file,
        or a pathlib.Path object pointing to a configuration file.
    is_public : bool, optional
        Whether the upload is meant to be public or not. This does not affect any real functionality
        and is soley used to sort the resulting NZB file. Default is False.
    debug : bool, optional
        Whether to enable debug logs. Default is False.

    Returns
    -------
    JuicenetOutput
        Dataclass used to represent the output of Juicenet.

    Raises
    ------
    ValueError
        Raised if `path` is not a string or pathlib.Path, or if it does not point to an existing file
        or `config` is not a string, a dictionary, or a pathlib.Path, or if it does not point to an existing file.

    Examples
    --------
    >>> from pathlib import Path
    >>> from juicenet import juicenet
    >>> file = Path("C:/Users/raven/Videos/Big Buck Bunny.mkv").resolve() # Recommended to always use resolved pathlib.Path
    >>> config = "D:/data/usenet/juicenetConfig/ENVjuicenet.yaml" # String also works, but not recommended
    >>> upload = juicenet(file, config=config)
    >>> upload.files[file].nyuu.nzb # You can access the nzb attribute to get the resolved pathlib.Path to the resulting nzb
    WindowsPath('D:/data/usenet/nzbs/private/Big Buck Bunny.mkv/Big Buck Bunny.mkv.nzb')
    >>> upload.files[file].nyuu.returncode # check the return code
    0
    >>> upload.files[file].parpar.par2files # Complete list of PAR2 files generated for the input file
    [WindowsPath('C:/Users/raven/AppData/Local/Temp/.JUICENET_i0dk8c_5/87540762A9/Big Buck Bunny.mkv.par2'), ...]
    >>> upload.files[file].parpar.filepathformat # ParPar `--filepath-format` used to generate the PAR2 files
    'basename'
    >>> upload.files[file].parpar.filepathbase # ParPar `--filepath-base` used to generate the PAR2 files
    WindowsPath('C:/Users/raven/Videos')
    """

    if isinstance(path, str):
        _path = Path(path).resolve()
    elif isinstance(path, Path):
        _path = path.resolve()
    else:
        raise ValueError("Path must be a string or pathlib.Path")

    if not _path.is_file():
        ValueError(f"{_path} must be an existing file")

    if isinstance(config, str):
        _config = Path(config).resolve()
    elif isinstance(config, Path):
        _config = config.resolve()
    elif isinstance(config, dict):
        _config = config  # type: ignore
    else:
        raise ValueError("Config must be a path or a dictonary")

    return main(path=_path, config=_config, no_resume=True, public=is_public, debug=debug)
