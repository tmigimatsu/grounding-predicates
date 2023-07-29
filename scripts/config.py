import pathlib


class EnvironmentPaths:
    """Class for accessing repository directories.

    Args:
        root (str, optional): Root of repository (default "..")
        environment (str, optional): Name of environment (default "gridworld")
    """
    def __init__(self, root="..", environment="gridworld"):
        self.root = pathlib.Path(root)
        self.environment = environment

    @property
    def data(self):
        """Data folder.

        :type: pathlib.Path
        """
        path = self.root / "data" / self.environment
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def env(self):
        """Environment folder.

        :type: pathlib.Path
        """
        path = self.root / "env" / self.environment
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models(self):
        """Trained network models folder.

        :type: pathlib.Path
        """
        path = self.root / "models" / self.environment
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def domain_pddl(self):
        """Domain pddl path.

        :type: pathlib.Path
        """
        return self.env / "domain.pddl"

    @property
    def problem_pddl(self):
        """Problem pddl path.

        :type: pathlib.Path
        """
        return self.env / "problem.pddl"
