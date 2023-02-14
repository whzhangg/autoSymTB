"""Module containing the custodian jobs for SISSO."""

import os
import subprocess
import shutil

from custodian import Custodian
from custodian.custodian import Job, Validator  # type: ignore


def runSISSO() -> None:
    """run job in the current folder"""
    job = SISSOJob()
    c = Custodian(jobs = [job], handlers=[], validators=[])
    c.run()


class SISSOJob(Job):
    """Custodian Job to run SISSO."""

    INPUT_FILE = "SISSO.in"
    TRAINING_DATA_DILE = "train.dat"

    def __init__(
        self,
        SISSO_exe: str = "SISSO",
        nprocs: int = 1,
        stdout_file: str = "SISSO.log",
        stderr_file: str = "SISSO.err",
    ):
        """Construct SISSOJob class.

        Args:
            SISSO_exe: Name of the SISSO executable.
            nprocs: Number of processors for the job.
            stdout_file: Name of the output file (default: SISSO.log).
            stderr_file: Name of the error file (default: SISSO.err).
        """
        self.SISSO_exe = SISSO_exe
        self.nprocs = nprocs
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file

    def setup(self):  # pragma: no cover
        """Not needed for SISSO."""
        pass

    def postprocess(self):
        """remove the (empty) error file"""
        if (os.path.isfile(self.stderr_file) and
                os.stat(self.stderr_file).st_size == 0):
            os.remove(self.stderr_file)

    def run(self) -> subprocess.Popen:
        """Run SISSO.

        Returns:
            a Popen process.
        """
        exe = shutil.which(self.SISSO_exe)
        if exe is None:
            raise ValueError(
                "SISSOJob requires the SISSO executable to be in the path.\n"
                'Default executable name is "SISSO" and you provided "{}".\n'
                "Download the SISSO code at https://github.com/rouyang2017/SISSO "
                "and compile the executable or fix the name of your executable.".format(
                    self.SISSO_exe
                )
            )

        if (self.nprocs > 1):
            raise NotImplementedError("Running SISSO with MPI not yet implemented.")
        else:
            cmd = exe

        with open(self.stdout_file, "w") as f_stdout, open(
            self.stderr_file, "w", buffering=1
        ) as f_stderr:
            p = subprocess.Popen(cmd, stdin=None, stdout=f_stdout, stderr=f_stderr)
        return p


class NormalCompletionValidator(Validator):
    """Validator of the normal completion of SISSO."""

    def __init__(
        self,
        output_file: str = "SISSO.out",
        stdout_file: str = "SISSO.log",
        stderr_file: str = "SISSO.err",
    ):
        """Construct NormalCompletionValidator class.

        This validator checks that the standard error file (SISSO.err by default) is
        empty, that the standard output file is not empty and that the output file
        (SISSO.out) is completed, i.e. ends with "Have a nice day !"

        Args:
            output_file: Name of the output file (default: SISSO.log).
            stdout_file: Name of the standard output file (default: SISSO.log).
            stderr_file: Name of the standard error file (default: SISSO.err).
        """
        self.output_file = output_file
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file

    def check(self) -> bool:
        """Validate the normal completion of SISSO.

        Returns:
            bool: True if the standard error file is empty, the standard output file
                is not empty and the output file ends with "Have a nice day !".
        """
        if not os.path.isfile(self.output_file):
            return True

        if not os.path.isfile(self.stdout_file):
            return True

        if os.stat(self.stdout_file).st_size == 0:
            return True

        if os.path.isfile(self.stderr_file):
            if os.stat(self.stderr_file).st_size != 0:
                return True

        with open(self.output_file, "rb") as f:
            out = f.read()

        return out.rfind(b"Have a nice day !") < 0
