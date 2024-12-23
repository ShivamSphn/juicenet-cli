import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from queue import Queue

from loguru import logger
from rich.progress import Progress

from .nyuu import Nyuu
from .parpar import ParPar
from .types import NyuuOutput, ParParOutput, SubprocessOutput
from .utils import get_related_files

# Thread-safe progress update queue
progress_queue: Queue = Queue()
# Lock for thread-safe logging
log_lock = threading.Lock()

def update_progress_safely(progress: Optional[Progress], task_id: Optional[str], advance: int = 1) -> None:
    """Thread-safe progress bar update"""
    if progress and task_id:
        try:
            progress_queue.put((progress, task_id, advance), block=False)
        except Exception as e:
            with log_lock:
                logger.error(f"Progress queue update failed: {str(e)}")


def process_progress_updates() -> None:
    """Process queued progress updates"""
    try:
        while not progress_queue.empty():
            try:
                progress, task_id, advance = progress_queue.get_nowait()
                try:
                    progress.update(task_id, advance=advance)
                except Exception as e:
                    with log_lock:
                        logger.error(f"Progress update failed: {str(e)}")
                finally:
                    progress_queue.task_done()
            except Exception:
                # Queue might become empty between check and get
                break
    except Exception as e:
        with log_lock:
            logger.error(f"Progress updates processing failed: {str(e)}")

def log_safely(level: str, message: str) -> None:
    """Thread-safe logging"""
    try:
        with log_lock:
            if level == "info":
                logger.info(message)
            elif level == "error":
                logger.error(message)
            elif level == "success":
                logger.success(message)
    except Exception as e:
        # Last resort logging if lock fails
        print(f"Logging failed: {level} - {message} - Error: {str(e)}")

def get_optimal_workers(max_workers: Optional[int] = None) -> int:
    """Calculate optimal number of worker threads"""
    if max_workers is not None:
        return max_workers
    try:
        cpu_count = threading.cpu_count()
        # Use CPU count + 4 for I/O bound tasks, but cap at 32
        return min(32, (cpu_count or 1) + 4)
    except Exception:
        # Fallback to safe default if CPU count fails
        return 8

def parallel_parpar(
    files: List[Path],
    parpar: ParPar,
    related_exts: List[str],
    progress: Optional[Progress] = None,
    task_id: Optional[str] = None,
    max_workers: int = None
) -> Dict[Path, ParParOutput]:
    """
    Generate par2 files for multiple files in parallel using ThreadPoolExecutor
    """
    results = {}
    
    def process_file(file: Path) -> Tuple[Path, ParParOutput]:
        try:
            related_files = get_related_files(file, exts=related_exts)
            if related_files:
                log_safely("info", f"Found {len(related_files)} related files for {file.name}")
            else:
                log_safely("info", f"No related files found for {file.name}")
                
            output = parpar.generate_par2_files(file, related_files=related_files)
            update_progress_safely(progress, task_id)
            return file, output
        except Exception as e:
            log_safely("error", f"ParPar processing failed for {file.name}: {str(e)}")
            # Return a failed output instead of raising to prevent thread crash
            return file, ParParOutput(
                par2files=[],
                success=False,
                filepathbase=file.parent,
                filepathformat="basename" if file.is_file() else "path",
                args=[],
                returncode=1,
                stdout="",
                stderr=f"Processing failed: {str(e)}"
            )

    with ThreadPoolExecutor(max_workers=get_optimal_workers(max_workers)) as executor:
        future_to_file = {
            executor.submit(process_file, file): file 
            for file in files
        }
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                file, output = future.result()
                results[file] = output
                if output.success:
                    log_safely("success", f"ParPar: {file.name}")
                else:
                    log_safely("error", f"ParPar failed: {file.name}")
            except Exception as e:
                log_safely("error", f"ParPar failed for {file.name}: {str(e)}")
                results[file] = ParParOutput(
                    par2files=[],
                    success=False,
                    filepathbase=file.parent,
                    filepathformat="basename" if file.is_file() else "path",
                    args=[],
                    returncode=1,
                    stdout="",
                    stderr=f"Future result failed: {str(e)}"
                )
                
        # Process any remaining progress updates
        process_progress_updates()
        return results

def parallel_nyuu(
    files: List[Path],
    nyuu: Nyuu,
    parpar_outputs: Dict[Path, ParParOutput],
    related_exts: List[str],
    progress: Optional[Progress] = None,
    task_id: Optional[str] = None,
    max_workers: int = None
) -> Dict[Path, NyuuOutput]:
    """
    Upload multiple files in parallel using ThreadPoolExecutor
    """
    results = {}
    
    def process_file(file: Path) -> Tuple[Path, NyuuOutput]:
        try:
            related_files = get_related_files(file, exts=related_exts)
            parpar_output = parpar_outputs.get(file)
            if not parpar_output:
                log_safely("error", f"No ParPar output found for {file.name}")
                return file, NyuuOutput(success=False, args=[], returncode=1, stdout="", stderr="No ParPar output found")
                
            output = nyuu.upload(
                file=file,
                related_files=related_files,
                par2files=parpar_output.par2files if parpar_output else []
            )
            update_progress_safely(progress, task_id)
            return file, output
        except Exception as e:
            log_safely("error", f"Nyuu processing failed for {file.name}: {str(e)}")
            return file, NyuuOutput(
                success=False,
                args=[],
                returncode=1,
                stdout="",
                stderr=f"Processing failed: {str(e)}"
            )

    with ThreadPoolExecutor(max_workers=get_optimal_workers(max_workers)) as executor:
        future_to_file = {
            executor.submit(process_file, file): file 
            for file in files
        }
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                file, output = future.result()
                results[file] = output
                if output.success:
                    log_safely("success", f"Nyuu: {file.name}")
                else:
                    log_safely("error", f"Nyuu failed: {file.name}")
            except Exception as e:
                log_safely("error", f"Nyuu failed for {file.name}: {str(e)}")
                results[file] = NyuuOutput(
                    success=False,
                    args=[],
                    returncode=1,
                    stdout="",
                    stderr=f"Future result failed: {str(e)}"
                )
                
        # Process any remaining progress updates
        process_progress_updates()
        return results

def process_files_parallel(
    files: List[Path],
    parpar: ParPar,
    nyuu: Nyuu,
    related_exts: List[str],
    progress: Optional[Progress] = None,
    parpar_task_id: Optional[str] = None,
    nyuu_task_id: Optional[str] = None,
    max_workers: int = None
) -> Dict[Path, SubprocessOutput]:
    """
    Process multiple files in parallel:
    1. First generate all par2 files in parallel
    2. Then upload all files in parallel
    """
    try:
        # First generate all par2 files in parallel
        parpar_outputs = parallel_parpar(
            files=files,
            parpar=parpar,
            related_exts=related_exts,
            progress=progress,
            task_id=parpar_task_id,
            max_workers=max_workers
        )
        
        # Filter out failed parpar outputs before nyuu
        successful_files = [
            file for file, output in parpar_outputs.items() 
            if output and output.success
        ]
        
        if not successful_files:
            log_safely("error", "No files successfully processed by ParPar")
            return {file: SubprocessOutput(parpar=output) for file, output in parpar_outputs.items()}
        
        # Then upload all files in parallel
        nyuu_outputs = parallel_nyuu(
            files=successful_files,
            nyuu=nyuu,
            parpar_outputs=parpar_outputs,
            related_exts=related_exts,
            progress=progress,
            task_id=nyuu_task_id,
            max_workers=max_workers
        )
    
        # Combine results
        results = {}
        for file in files:
            results[file] = SubprocessOutput(
                parpar=parpar_outputs.get(file),
                nyuu=nyuu_outputs.get(file) if file in successful_files else None
            )
            
        return results
    except Exception as e:
        log_safely("error", f"Parallel processing failed: {str(e)}")
        # Return empty results rather than crashing
        return {file: SubprocessOutput() for file in files}
