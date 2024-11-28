"""
FastAPI implementation of Juicenet for Usenet uploads.
This module provides a RESTful API interface for the core Juicenet functionality,
allowing users to upload files to Usenet, manage configurations, and perform file operations.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

from rich.console import Console
from rich.traceback import install

from ..config import get_dump_failed_posts, read_config
from ..exceptions import JuicenetInputError
from ..model import JuicenetConfig
from ..nyuu import Nyuu
from ..parpar import ParPar
from ..resume import Resume
from ..types import JuiceBox, NyuuOutput, ParParOutput, StrPath
from ..utils import filter_empty_files, get_glob_matches, get_related_files, get_bdmv_discs

# Install rich traceback
install()

# Console object, used by both progressbar and loguru
console = Console()

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Juicenet API",
    description="""
    FastAPI implementation of Juicenet for Usenet uploads.
    Provides endpoints for file uploads, configuration management, and file operations.
    """,
    version="1.0.0"
)

# API Tags for documentation organization
tags_metadata = [
    {
        "name": "upload",
        "description": "Operations related to Usenet uploads"
    },
    {
        "name": "files",
        "description": "File system operations and utilities"
    },
    {
        "name": "config",
        "description": "Configuration management"
    },
    {
        "name": "system",
        "description": "System health and status"
    }
]

class UploadRequest(BaseModel):
    """
    Request model for file upload operations.
    
    Attributes:
        path (str): Path to the file or directory to upload
        config (Union[str, JuicenetConfig]): Path to config file or JuicenetConfig instance
        public (bool): Whether to use public upload settings (default: False)
        bdmv_naming (bool): Whether to use BDMV naming convention (default: False)
        resume (bool): Whether to enable upload resumption (default: True)
        skip_raw (bool): Whether to skip raw article uploads (default: False)
        debug (bool): Whether to enable debug logging (default: False)
    """
    path: str = Field(..., description="Path to the file or directory to upload")
    config: Union[str, JuicenetConfig] = Field(..., description="Path to config file or JuicenetConfig instance")
    public: bool = Field(False, description="Whether to use public upload settings")
    bdmv_naming: bool = Field(False, description="Whether to use BDMV naming convention")
    resume: bool = Field(True, description="Whether to enable upload resumption")
    skip_raw: bool = Field(False, description="Whether to skip raw article uploads")
    debug: bool = Field(False, description="Whether to enable debug logging")

    class Config:
        arbitrary_types_allowed = True

class FileListRequest(BaseModel):
    """
    Request model for file listing operations.
    
    Attributes:
        path (str): Directory path to search for files
        exts (List[str]): List of file extensions to match (default: ["mkv"])
    """
    path: str = Field(..., description="Directory path to search for files")
    exts: List[str] = Field(default=["mkv"], description="List of file extensions to match")

class GlobMatchRequest(BaseModel):
    """
    Request model for glob pattern matching operations.
    
    Attributes:
        path (str): Directory path to search in
        globs (List[str]): List of glob patterns to match (default: ["*.mkv"])
    """
    path: str = Field(..., description="Directory path to search in")
    globs: List[str] = Field(default=["*.mkv"], description="List of glob patterns to match")

class BDMVRequest(BaseModel):
    """
    Request model for BDMV disc detection.
    
    Attributes:
        path (str): Directory path to search for BDMV discs
        globs (List[str]): List of glob patterns to match (default: ["*/"])
    """
    path: str = Field(..., description="Directory path to search for BDMV discs")
    globs: List[str] = Field(default=["*/"], description="List of glob patterns to match")

class ConfigRequest(BaseModel):
    """
    Request model for configuration validation.
    
    Attributes:
        config_path (str): Path to the configuration file to validate
    """
    config_path: str = Field(..., description="Path to the configuration file to validate")

@app.post("/upload", response_model=JuiceBox, 
         tags=["upload"],
         summary="Upload file to Usenet",
         description="Upload a file or folder to Usenet, producing one NZB file for one input")
async def upload_file(request: UploadRequest) -> JuiceBox:
    """
    Upload a file or folder to Usenet.
    
    This endpoint handles the complete upload process including:
    - File validation and processing
    - Configuration management
    - PAR2 file generation
    - Usenet upload via Nyuu
    - Resume functionality
    
    Args:
        request (UploadRequest): Upload request containing path and configuration details
        
    Returns:
        JuiceBox: Object containing upload results including NZB file info and status
        
    Raises:
        HTTPException: 
            - 404 if file not found
            - 400 for invalid input or empty files
            - 500 for internal server errors
            
    Notes:
        - You should never upload an entire directory consisting of several files as a single NZB.
          Use the /files or /glob endpoints to first get the relevant files and then pass each one
          to this endpoint.
        - You should never upload an entire BDMV consisting of several discs as a single NZB.
          Use the /bdmv endpoint to first get each individual disc and then pass each one to this endpoint.
    """
    try:
        # Resolve and validate input path
        _path = Path(request.path).resolve()
        
        if not _path.exists():
            raise HTTPException(status_code=404, detail=f"{_path} must be an existing file or directory")

        # Check for empty files
        filelist = filter_empty_files([_path])
        if len(filelist) == 1:
            file = filelist[0]
        else:
            raise HTTPException(status_code=400, detail=f"{_path} is empty (0-byte)!")

        # Handle config which can be either a path or JuicenetConfig
        if isinstance(request.config, str):
            _config = Path(request.config).resolve()
        elif isinstance(request.config, JuicenetConfig):
            _config = request.config
        else:
            raise HTTPException(status_code=400, detail="Config must be a path or a juicenet.JuicenetConfig")

        # Load and validate configuration
        config_data = read_config(_config)

        # Extract configuration values
        nyuu_bin = config_data.nyuu
        parpar_bin = config_data.parpar
        priv_conf = config_data.nyuu_config_private
        pub_conf = config_data.nyuu_config_public or priv_conf
        nzb_out = config_data.nzb_output_path
        parpar_args = config_data.parpar_args
        related_exts = config_data.related_extensions

        # Setup application directories
        appdata_dir = config_data.appdata_dir_path
        appdata_dir.mkdir(parents=True, exist_ok=True)
        resume_file = appdata_dir / "juicenet.resume"
        resume_file.touch(exist_ok=True)

        # Configure working directory
        work_dir = config_data.temp_dir_path if config_data.use_temp_dir else None

        # Select appropriate configuration based on public/private setting
        configurations = {"public": pub_conf, "private": priv_conf}
        scope = "public" if request.public else "private"
        conf = configurations[scope]

        # Handle raw articles if present
        dump = get_dump_failed_posts(conf)
        raw_articles = get_glob_matches(dump, ["*"])
        raw_count = len(raw_articles)

        # Initialize resume functionality
        no_resume = not request.resume
        _resume = Resume(resume_file, scope, no_resume)

        # Initialize PAR2 generator
        parpar = ParPar(parpar_bin, parpar_args, work_dir, request.debug)

        # Configure BDMV naming
        bdmv_naming = request.bdmv_naming and not file.is_file()

        # Initialize Nyuu uploader
        nyuu = Nyuu(file.parent.parent, nyuu_bin, conf, work_dir, nzb_out, scope, request.debug, bdmv_naming)

        # Process raw articles if needed
        rawoutput = {}
        if raw_count and (not request.skip_raw):
            for article in raw_articles:
                raw_out = nyuu.repost_raw(article=article)
                rawoutput[article] = raw_out

        # Check if file was already uploaded
        if _resume.already_uploaded(file):
            return JuiceBox(
                nyuu=NyuuOutput(nzb=None, success=False, args=[], returncode=1, stdout="", stderr=""),
                parpar=ParParOutput(
                    par2files=[],
                    success=False,
                    filepathbase=file.parent,
                    filepathformat="basename" if file.is_file() else "path",
                    args=[],
                    returncode=1,
                    stdout="",
                    stderr="",
                ),
                raw={},
                skipped=True,
            )

        # Process new upload
        related_files = get_related_files(file, exts=related_exts) if file.is_file() else None
        parpar_out = parpar.generate_par2_files(file, related_files=related_files)
        nyuu_out = nyuu.upload(file=file, related_files=related_files, par2files=parpar_out.par2files)

        # Log successful upload
        if nyuu_out.success:
            _resume.log_file_info(file)

        return JuiceBox(nyuu=nyuu_out, parpar=parpar_out, raw=rawoutput, skipped=False)

    except JuicenetInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/files", response_model=List[str],
          tags=["files"],
          summary="List files by extension",
          description="Get a recursive list of files with specified extensions from the given path")
async def get_files_endpoint(request: FileListRequest) -> List[str]:
    """
    Get a list of files with specified extensions from the given path.
    
    This endpoint recursively searches a directory for files matching the specified extensions.
    
    Args:
        request (FileListRequest): Request containing path and file extensions to match
        
    Returns:
        List[str]: List of matching file paths
        
    Raises:
        HTTPException: 
            - 400 if path is not a directory
            - 500 for internal server errors
    """
    try:
        basepath = Path(request.path).resolve()
        
        if not basepath.is_dir():
            raise HTTPException(status_code=400, detail=f"{basepath} must be a directory")

        # Collect files matching each extension
        files = []
        for ext in request.exts:
            matches = list(basepath.rglob(f"*.{ext.strip('.')}"))
            files.extend(matches)

        return [str(f) for f in files]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/glob", response_model=List[str],
          tags=["files"],
          summary="Match files using glob patterns",
          description="Get a list of files matching the specified glob patterns in the given path")
async def get_glob_matches_endpoint(request: GlobMatchRequest) -> List[str]:
    """
    Get a list of files matching glob patterns in the specified path.
    
    This endpoint searches a directory for files matching the provided glob patterns.
    
    Args:
        request (GlobMatchRequest): Request containing path and glob patterns to match
        
    Returns:
        List[str]: List of matching file paths
        
    Raises:
        HTTPException: 
            - 400 if path is not a directory
            - 500 for internal server errors
    """
    try:
        basepath = Path(request.path).resolve()
        
        if not basepath.is_dir():
            raise HTTPException(status_code=400, detail=f"{basepath} must be a directory")

        # Collect files matching each glob pattern
        files = []
        for glob in request.globs:
            matches = list(basepath.glob(glob))
            files.extend(matches)

        return [str(f) for f in files]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bdmv", response_model=List[str],
          tags=["files"],
          summary="Detect BDMV discs",
          description="Find individual discs in BDMVs by looking for BDMV/index.bdmv")
async def get_bdmv_discs_endpoint(request: BDMVRequest) -> List[str]:
    """
    Find individual discs in BDMVs by looking for BDMV/index.bdmv.
    
    This endpoint searches for BDMV disc structures in the specified directory.
    
    Args:
        request (BDMVRequest): Request containing path and glob patterns to match
        
    Returns:
        List[str]: List of paths to BDMV discs
        
    Raises:
        HTTPException: 
            - 404 if path doesn't exist
            - 500 for internal server errors
    """
    try:
        basepath = Path(request.path).resolve()
        
        if not basepath.exists():
            raise HTTPException(status_code=404, detail=f"{basepath} must be an existing directory")

        # Find BDMV discs
        bdmvs = get_bdmv_discs(basepath, request.globs)
        return [str(f) for f in bdmvs]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/validate", response_model=Dict[str, Union[str, bool, dict]],
          tags=["config"],
          summary="Validate configuration",
          description="Validate a Juicenet configuration file")
async def validate_config(request: ConfigRequest) -> Dict[str, Union[str, bool, dict]]:
    """
    Validate a Juicenet configuration file.
    
    This endpoint attempts to read and validate a configuration file,
    returning success status and any validation errors.
    
    Args:
        request (ConfigRequest): Request containing path to config file
        
    Returns:
        Dict[str, Union[str, bool, dict]]: Validation result with status and any error messages
        
    Raises:
        HTTPException: 
            - 404 if config file not found
            - 500 for internal server errors
    """
    try:
        config_path = Path(request.config_path).resolve()
        
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file {config_path} not found")

        try:
            config = read_config(config_path)
            return {
                "valid": True,
                "message": "Configuration is valid",
                "config": config.model_dump()
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Configuration validation failed: {str(e)}"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health",
         tags=["system"],
         summary="Health check",
         description="Check if the API is running and healthy")
async def health_check():
    """
    Simple health check endpoint to verify API status.
    
    Returns:
        dict: Status information
    """
    return {"status": "healthy"}
