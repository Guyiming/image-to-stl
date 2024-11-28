from pydantic import BaseModel, Field
from typing import Dict, Optional
import yaml
from pathlib import Path
from Models import Filament


class FilamentLibrary(BaseModel):
    filaments: dict[str, Filament]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FilamentLibrary":
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(filaments=data)

    def get_filament(self, filament_id: str) -> Optional[Filament]:
        return self.filaments.get(filament_id)